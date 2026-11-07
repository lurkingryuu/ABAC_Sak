(() => {
    const assets = new URL('demo/', document.currentScript.src);
    const asset = (name) => new URL(name, assets).href;
    const $ = (id) => document.getElementById(id);
    const steps = [
        [0, 'Configure a synthetic ABAC benchmark', ''],
        [20, 'Describe an attribute with a sketch', 'Keep the entity sizes and rule counts in JSON. The sketches supply attribute definitions and distributions.'],
        [36, 'Check the fit and generated counts', 'Compare what MuSimA read from the drawing with the values it assigned to subjects.'],
        [58, 'Trace one access decision', 'Start with three entity IDs. Look up their assigned attribute values, match the rule, and read the resulting decision in the access control matrix.'],
        [74, 'Sample a synthetic access log', 'Generate Logs samples requests from the matrix. The requested permit share controls the CSV sample. It does not change the policy or the matrix.'],
    ];
    const entityKeys = ['subject_size', 'object_size', 'environment_size'];
    const ruleKeys = ['permit_rules_count', 'deny_rules_count'];
    const configFields = [
        [entityKeys, '200 subjects × 30 objects × 3 environment instances gives 18,000 possible requests.'],
        [['subject_attributes_count', 'subject_attributes_values'], 'One subject attribute, SA_1, has 12 possible values. Each subject is assigned one of them.'],
        [['subject_distributions'], 'N means Normal. A mean of 4 centers the distribution near SA_1_4. Variance controls how spread out the assignments are.'],
        [ruleKeys, 'MuSimA generates five permit rules and three deny rules. The rules evaluate attribute values to decide each request.'],
    ];

    async function load(name, read) {
        const response = await fetch(asset(name));
        if (!response.ok) throw new Error('The tutorial example could not load.');
        return response[read]();
    }

    function setFiles(input, files) {
        const transfer = new DataTransfer();
        files.forEach((file) => transfer.items.add(file));
        input.files = transfer.files;
        input.dispatchEvent(new Event('change', { bubbles: true }));
    }

    function waitForEditors(workspace) {
        return new Promise((resolve, reject) => {
            const deadline = Date.now() + 12000;
            function check() {
                const editors = workspace.editors();
                if (editors.every(Boolean)) resolve(editors);
                else if (Date.now() >= deadline) reject(new Error('The workspace editor is still loading. Try starting the tutorial again.'));
                else setTimeout(check, 100);
            }
            check();
        });
    }

    function node(tag, text) {
        const element = document.createElement(tag);
        if (text !== undefined) element.textContent = text;
        return element;
    }

    function chart(run, max) {
        const figure = node('figure');
        const svg = (tag, attrs, text) => {
            const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
            for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
            if (text !== undefined) element.textContent = text;
            return element;
        };
        const plot = svg('svg', {
            viewBox: '0 0 306 145', role: 'img',
            'aria-label': run.bins.map(bin => `Value ${bin.level}: ${bin.observed} observed, ${bin.expected} expected`).join('. '),
        });
        const width = 266 / run.bins.length;
        const x = (index) => 30 + width * (index + 0.5);
        const y = (value) => 108 - 86 * value / max;
        plot.append(svg('text', { x: 5, y: 13 }, 'Subjects'), svg('text', { x: 5, y: 26 }, max),
            svg('text', { x: 12, y: 110 }, '0'), svg('text', { x: 155, y: 143, 'text-anchor': 'middle' }, 'SA_1 value index'));
        run.bins.forEach((bin, index) => {
            plot.append(svg('rect', { x: x(index) - width * .34, y: y(bin.observed), width: width * .68, height: 108 - y(bin.observed) }),
                svg('text', { x: x(index), y: 124, 'text-anchor': 'middle' }, bin.level));
        });
        plot.append(svg('polyline', { points: run.bins.map((bin, index) => `${x(index)},${y(bin.expected)}`).join(' ') }));
        figure.append(node('figcaption', `${run.label} · ${run.family}`), plot);
        return figure;
    }

    function exampleComparison(data) {
        const fragment = document.createDocumentFragment();
        const heading = node('h3', 'Distribution attestation · Two example runs');
        const fit = node('pre');
        fit.id = 'tutorial-fitted-json';
        fit.className = 'tutorial-fit-json';
        const { subject_attributes_values, subject_distributions } = data.standard;
        const fitCode = node('code');
        const fittedJson = JSON.stringify({ subject_attributes_values, subject_distributions }, null, 2);
        fittedJson.split('\n').forEach((line, index) => {
            if (index) fitCode.append('\n');
            const row = node('span', line);
            if (/^\s*(?:"|\d)/.test(line)) row.className = 'tutorial-config-line';
            fitCode.append(row);
        });
        fit.append(fitCode);
        const figure = node('figure');
        figure.id = 'tutorial-fit-comparison';
        const image = node('img');
        image.src = asset('peaked-fit.png');
        image.alt = 'Original SA-1 sketch beside the Normal curve Gemini fitted';
        figure.append(image, node('figcaption', 'Original hand drawing and the fitted curve from the same run.'));
        const charts = node('div');
        charts.className = 'tutorial-charts';
        charts.id = 'tutorial-counts';
        const max = Math.ceil(Math.max(...data.runs.flatMap(run => run.bins.flatMap(bin => [bin.expected, bin.observed]))) / 10) * 10;
        charts.append(...data.runs.map(run => chart(run, max)));
        fragment.append(heading, fit, figure, node('p', 'Bars show generated counts. The dashed line shows expected counts. Both plots use the same scale.'), charts);
        return fragment;
    }

    function exampleTrace({ entities, values, rule, matrix, flattened, csv }) {
        const [subject, resource, environment] = entities;
        const [sa, oa, ea] = values;
        const fragment = document.createDocumentFragment();
        const list = node('dl');
        list.className = 'tutorial-trace';
        const rows = [
            ['1. Identify the request', `(${entities.join(', ')})`, 'Subject, object, environment instance. Each is an entity ID.'],
            ['2. Look up assigned values', `SV[${subject}] → ${sa} · OV[${resource}] → ${oa} · EV[${environment}] → ${ea}`, `SA_1, OA_1, and EA_1 are the first subject, object, and environment attributes. ${sa} is value ${sa.split('_').at(-1)} of SA_1. SAV lists possible values, and SV gives each subject’s assignments.`],
            ['3. Evaluate the rules', rule, 'All three conditions match this permit rule, and no deny rule matches. When both types are present, a matching deny takes precedence.'],
            ['4. Locate the decision', `ACM[${entities.join(', ')}] = ${matrix}`, '1 means permit. The flattened dataset records the assigned values and decision together.'],
            ['5. Match the exported rows', flattened, `The sampled CSV uses entity IDs: ${csv}.`],
        ];
        for (const [title, code, explanation] of rows) {
            const item = node('div');
            const detail = node('dd');
            detail.append(node('code', code), node('p', explanation));
            item.append(node('dt', title), detail);
            list.append(item);
        }
        fragment.append(node('h3', 'Example request from the peaked run'), list);
        return fragment;
    }

    function mount(workspace) {
        const panel = $('tutorial');
        const output = $('tutorial-output');
        const start = $('tutorial-start');
        const zipInput = $('zip-input');
        const progress = $('progress-container');
        const download = $('download-container');
        const sketch = $('tutorial-sketch');
        const timeline = $('tutorial-progress');
        const note = $('tutorial-note');
        const fieldButtons = [...$('tutorial-fields').querySelectorAll('button')];
        const autoplayControls = ['tutorial-play', 'tutorial-restart', 'tutorial-progress'];
        const replayInputs = [$('editor-tab'), $('multimodal-form'), document.querySelector('.log-generator-container'), document.querySelector('.tabs')];
        const inputModes = ['editor', 'multimodal'];
        const collapsed = () => [document.querySelector('#editor-tab .editor-wrapper'), $('multimodal-form')]
            .map((element) => element.classList.contains('is-collapsed'));
        const duration = Number(timeline.max);
        const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
        let example;
        let sampleZip;
        let session;
        let replay;
        let starting = false;
        let current = 0;
        let phase = -1;
        let highlight;
        let liveRun = false;
        let elapsed = 0;
        let origin = 0;
        let playing = false;
        let frame;
        let fieldHighlight;

        function setNote(message) {
            note.textContent = message;
            note.hidden = !message;
        }

        function setCaption(message) {
            $('tutorial-caption').textContent = message;
        }

        function spotlight(element) {
            highlight?.classList.remove('tutorial-focus');
            highlight = element;
            if (!highlight) return;
            highlight.classList.add('tutorial-focus');
            const viewport = highlight.closest('.content-scroll');
            const block = viewport && highlight.clientHeight > viewport.clientHeight ? 'start' : 'nearest';
            highlight.scrollIntoView({ block, behavior: 'instant' });
        }

        function clearFields() {
            fieldHighlight?.clear();
            fieldHighlight = null;
        }

        function spotlightFields(index, keys) {
            clearFields();
            const editor = session.editors[index];
            const model = editor.getModel();
            const ranges = keys.flatMap(key => {
                const [match] = model.findMatches('"' + key + '"', false, false, true, null, false, 1);
                if (!match) return [];
                const first = match.range.startLineNumber;
                const line = model.getLineContent(first);
                let last = first;
                if (line.trimEnd().endsWith('[')) {
                    const indent = line.search(/\S/);
                    while (last < model.getLineCount()) {
                        last++;
                        const nextIndent = model.getLineContent(last).search(/\S/);
                        if (nextIndent >= 0 && nextIndent <= indent) break;
                    }
                }
                return [new monaco.Range(first, 1, last, model.getLineMaxColumn(last))];
            }).sort((a, b) => a.startLineNumber - b.startLineNumber);
            if (!ranges.length) return;
            fieldHighlight = editor.createDecorationsCollection(ranges.map(range => ({
                range, options: { isWholeLine: true, className: 'tutorial-config-line' },
            })));
            editor.layout();
            editor.revealLinesInCenterIfOutsideViewport(ranges[0].startLineNumber, ranges.at(-1).endLineNumber);
        }

        function showConfigField(index) {
            const [keys, caption] = configFields[index];
            setCaption(caption);
            fieldButtons.forEach((button, i) => button.setAttribute('aria-pressed', String(i === index)));
            spotlight($('json-editor'));
            spotlightFields(0, keys);
        }

        function setExampleInputs() {
            session.models[1].setValue(JSON.stringify(example.minimal, null, 2));
            setFiles(zipInput, [sampleZip]);
        }

        function updateControls() {
            $('tutorial-back').disabled = current === 0;
            $('tutorial-next').disabled = Boolean(!replay && current === 1 && !liveRun);
            $('tutorial-next').textContent = current === steps.length - 1 ? 'Finish' : 'Next';
            $('tutorial-next').hidden = Boolean(replay && current === steps.length - 1);
            $('tutorial-load').hidden = current !== 1 || Boolean(replay);
            $('tutorial-logs').hidden = current !== 4;
            $('tutorial-mode').textContent = replay ? 'Hands-on' : 'Autoplay';
            for (const id of autoplayControls) $(id).hidden = !replay;
        }

        function showOutput(index) {
            output.hidden = false;
            output.replaceChildren(index === 2 ? exampleComparison(example) : exampleTrace(example.trace));
            spotlight(output);
        }

        function showStage(index) {
            current = index;
            phase = -1;
            clearFields();
            $('tutorial-fields').hidden = index !== 0;
            output.hidden = true;
            sketch.hidden = true;
            const tab = ['editor', 'multimodal', 'multimodal', 'multimodal', 'log-gen'][index];
            if (replay && index === 4) workspace.showExampleResult(asset('peaked-output.zip'));
            workspace.selectTab(tab);
            const [, heading, caption] = steps[index];
            $('tutorial-step').textContent = `${index + 1} of ${steps.length} · ${replay ? 'Autoplay · Example run' : 'Hands-on tutorial'}`;
            $('tutorial-heading').textContent = heading;
            setCaption(caption);
            setNote('');
            updateControls();

            if (index < 2) {
                workspace.expandInput(tab);
                if (replay) {
                    workspace.clearExampleResult();
                    progress.style.display = 'none';
                    download.style.display = 'none';
                    if (index === 1) setFiles(zipInput, []);
                }
                if (index === 1 && !replay) setNote('Load the sample or use your own inputs. Complete verification and select Process Multimodal Input to continue.');
                if (index === 0 && !replay) showConfigField(0);
                if (index === 1 && !replay) {
                    spotlight($('min-json-editor'));
                    spotlightFields(1, [...entityKeys, ...ruleKeys]);
                }
            } else if (index === 2 && replay) {
                download.style.display = 'none';
                workspace.collapseInput('multimodal');
                progress.style.display = 'block';
                setNote('Showing a completed example with shortened progress.');
                spotlight(progress);
            } else if (index === 2 || index === 3) {
                if (replay) workspace.showExampleResult(asset('peaked-output.zip'));
                setNote(replay ? '' : 'The download button opens your run. The inspector shows the paired tutorial example.');
                showOutput(index);
            } else {
                if (replay) {
                    $('num-logs').value = 1000;
                    $('allow-percentage').value = 20;
                }
                setNote(`Example CSV row from the peaked run: ${example.trace.csv}`);
                spotlight($('generate-logs-btn'));
            }
        }

        function showPhase(seconds) {
            if (current === 0) {
                const next = Math.min(3, Math.floor(seconds / 5));
                if (phase === next) return;
                phase = next;
                showConfigField(next);
            } else if (current === 1) {
                const next = seconds < 4 ? 0 : seconds < 6 ? 1 : seconds < 13 ? 2 : 3;
                if (phase === next) return;
                phase = next;
                clearFields();
                if (next > 0 && !zipInput.files.length) setFiles(zipInput, [sampleZip]);
                sketch.hidden = next < 2;
                setCaption([
                    steps[1][2],
                    'The sample ZIP contains original drawings for a subject, an object, and an environment attribute.',
                    'Find SA-1 on the paper. It identifies the first subject attribute. The 12 at the right of the x-axis gives its number of possible values.',
                    'Process Multimodal Input reads the drawings, builds the full configuration, and runs the simulation.',
                ][next]);
                spotlight($(['min-json-editor', 'zip-wrapper', 'tutorial-sketch', 'submit-multimodal-btn'][next]));
                if (next === 0) spotlightFields(1, [...entityKeys, ...ruleKeys]);
            } else if (current === 2) {
                const next = seconds < 1.5 ? 0 : seconds < 3 ? 1 : seconds < 4.5 ? 2 : seconds < 10 ? 3 : seconds < 16 ? 4 : 5;
                if (phase === next) return;
                phase = next;
                workspace.resetProgressSteps();
                workspace.activateStep(['step-upload', 'step-process', 'step-generate', 'step-complete'][Math.min(next, 3)]);
                if (next >= 3 && output.hidden) {
                    workspace.showExampleResult(asset('peaked-output.zip'));
                    showOutput(2);
                }
                if (next >= 3) {
                    setCaption([
                        'MuSimA read 12 possible SA_1 values from the drawing and fitted a Normal distribution with mean 4 and variance 6.25.',
                        'The original hand drawing is on the left. On the right is the curve MuSimA fitted to it.',
                        'Bars count the subjects assigned each value. The dashed line shows the expected counts. Both runs use the same settings except for the SA_1 distribution.',
                    ][next - 3]);
                    spotlight($(['tutorial-fitted-json', 'tutorial-fit-comparison', 'tutorial-counts'][next - 3]));
                }
            } else if (current === 3) {
                const next = seconds < 5 ? 0 : seconds < 9 ? 2 : 4;
                if (phase === next) return;
                phase = next;
                spotlight(output.querySelector('.tutorial-trace').children[next]);
            }
        }

        function renderTime(time, refresh = false) {
            elapsed = time;
            timeline.value = elapsed;
            const index = steps.findLastIndex(([time]) => time <= elapsed);
            if (refresh || current !== index) showStage(index);
            showPhase(elapsed - steps[index][0]);
        }

        function tick(now) {
            renderTime(Math.max(0, (now - origin) / 1000) % duration);
            frame = requestAnimationFrame(tick);
        }

        function setPlaying(value) {
            cancelAnimationFrame(frame);
            playing = value;
            $('tutorial-play').textContent = playing ? 'Pause' : 'Resume';
            if (playing) {
                origin = performance.now() - elapsed * 1000;
                frame = requestAnimationFrame(tick);
            }
        }

        function stopReplay() {
            if (!replay) return;
            setPlaying(false);
            document.body.classList.remove('tutorial-autoplay');
            replayInputs.forEach((element) => { element.inert = false; });
            workspace.selectTab('editor');
            workspace.restoreResult(replay.result);
            $('num-logs').value = replay.logCount;
            $('allow-percentage').value = replay.allowShare;
            replay = null;
        }

        function startReplay() {
            if (workspace.busy()) {
                setNote('Wait for the current job to finish before starting autoplay.');
                return;
            }
            replay = { result: workspace.snapshotResult(), logCount: $('num-logs').value, allowShare: $('allow-percentage').value };
            replayInputs.forEach((element) => { element.inert = true; });
            document.body.classList.add('tutorial-autoplay');
            session.models[0].setValue(JSON.stringify(example.standard, null, 2));
            session.models[1].setValue(JSON.stringify(example.minimal, null, 2));
            renderTime(0, true);
            setPlaying(!reducedMotion);
        }

        function clearAutoplayLink() {
            const url = new URL(location.href);
            if (!url.searchParams.has('tutorial')) return;
            url.searchParams.delete('tutorial');
            history.replaceState(history.state, '', url);
        }

        function close() {
            starting = false;
            stopReplay();
            clearFields();
            if (session) {
                for (const [index, editor] of session.editors.entries()) {
                    editor.setModel(session.originals[index].model);
                    if (session.originals[index].view) editor.restoreViewState(session.originals[index].view);
                    session.models[index].dispose();
                }
                setFiles(zipInput, session.files);
                inputModes.forEach((mode, index) => {
                    if (session.collapsed[index]) workspace.collapseInput(mode);
                    else workspace.expandInput(mode);
                });
                workspace.selectTab(session.tab);
            }
            session = null;
            panel.hidden = true;
            start.hidden = false;
            output.hidden = true;
            sketch.hidden = true;
            output.replaceChildren();
            document.body.classList.remove('tutorial-active');
            spotlight(null);
            clearAutoplayLink();
            start.focus();
        }

        async function open(auto) {
            if (session || starting) return;
            if (workspace.busy()) {
                start.textContent = 'Wait for the current job, then start tutorial';
                return;
            }
            starting = true;
            panel.hidden = false;
            start.hidden = true;
            document.body.classList.add('tutorial-active');
            $('tutorial-fields').hidden = true;
            $('tutorial-heading').textContent = 'Loading tutorial';
            $('tutorial-caption').textContent = 'Preparing the existing workspace and example files.';
            $('tutorial-step').textContent = '';
            setNote('');
            panel.querySelector('.tutorial-controls').hidden = true;
            const attempt = Symbol();
            open.attempt = attempt;
            try {
                const [editors, data, zip] = await Promise.all([
                    waitForEditors(workspace), example ?? load('tutorial.json', 'json'), sampleZip ?? load('peaked-input.zip', 'blob'),
                ]);
                if (!starting || open.attempt !== attempt) return;
                if (workspace.busy()) throw new Error('Wait for the current job to finish, then start again.');
                example = data;
                sampleZip = zip instanceof File ? zip : new File([zip], 'peaked-input.zip', { type: 'application/zip' });
                sketch.querySelector('img').src = asset('peaked-sketch.jpg');
                const originals = editors.map(editor => ({ model: editor.getModel(), view: editor.saveViewState() }));
                const models = [example.standard, example.minimal].map(config => monaco.editor.createModel(JSON.stringify(config, null, 2), 'json'));
                session = { editors, originals, models, tab: document.querySelector('.tab-btn.active').dataset.tab,
                    files: [...zipInput.files], collapsed: collapsed() };
                editors.forEach((editor, index) => editor.setModel(models[index]));
                setFiles(zipInput, []);
                liveRun = false;
                panel.querySelector('.tutorial-controls').hidden = false;
                if (auto) startReplay();
                else showStage(0);
                $('tutorial-heading').focus({ preventScroll: true });
            } catch (error) {
                if (starting && open.attempt === attempt) {
                    $('tutorial-heading').textContent = 'Tutorial unavailable';
                    $('tutorial-caption').textContent = error.message;
                }
            } finally {
                if (open.attempt === attempt) starting = false;
            }
        }

        function advance(offset) {
            if (!replay) {
                if (current === steps.length - 1 && offset > 0) close();
                else showStage(current + offset);
                return;
            }
            const index = current + offset;
            const delay = playing ? 0 : [0, 6, 4.5, 0, 0][index];
            renderTime(steps[index][0] + delay, true);
            origin = performance.now() - elapsed * 1000;
        }

        start.hidden = false;
        start.addEventListener('click', () => open(false));
        $('tutorial-exit').addEventListener('click', close);
        $('tutorial-back').addEventListener('click', () => advance(-1));
        $('tutorial-next').addEventListener('click', () => advance(1));
        fieldButtons.forEach((button, index) => button.addEventListener('click', () => {
            if (replay) {
                setPlaying(false);
                renderTime(index * 5);
            } else {
                workspace.selectTab('editor');
                workspace.expandInput('editor');
                showConfigField(index);
            }
        }));
        $('tutorial-load').addEventListener('click', () => {
            setExampleInputs();
            sketch.hidden = false;
            clearFields();
            spotlight(sketch);
            setCaption('Find SA-1 and 12 on the drawing. They identify the first subject attribute and its number of possible values.');
            setNote('Example inputs loaded. Complete verification, then select Process Multimodal Input.');
        });
        $('tutorial-mode').addEventListener('click', () => {
            if (!replay) startReplay();
            else {
                stopReplay();
                showStage(0);
            }
        });
        $('tutorial-play').addEventListener('click', () => setPlaying(!playing));
        $('tutorial-restart').addEventListener('click', () => {
            renderTime(0, true);
            setPlaying(!reducedMotion);
        });
        for (const link of [$('download-bundle-link'), $('tutorial-logs')]) {
            link.addEventListener('focus', () => { if (replay) setPlaying(false); });
        }
        document.addEventListener('musima:job-completed', (event) => {
            if (!session || replay || event.detail.input !== 'multimodal') return;
            liveRun = true;
            if (current === 1) showStage(2);
        });
        if (new URL(location.href).searchParams.get('tutorial') === 'autoplay') open(true);
    }

    window.MuSimATutorial = { mount };
})();
