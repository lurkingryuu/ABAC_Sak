(() => {
    const $ = (id) => document.getElementById(id);
    const data = window.MUSIMA_DEMO;
    const tabs = [...document.querySelectorAll('.tab-btn')];
    const chapters = [...document.querySelectorAll('.demo-controls nav button')];
    const scroller = document.querySelector('.demo-workspace .content-scroll');
    const toggle = $('demo-toggle');
    const timeline = $('demo-timeline');
    const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

    if (!data) {
        $('demo-caption').textContent = 'The example could not load. Open the live workspace to try MuSimA.';
        document.querySelector('.demo-controls').hidden = true;
        return;
    }

    for (const count of document.querySelectorAll('[data-demo-count]')) {
        count.textContent = data.config[count.dataset.demoCount];
    }
    const [flat, peaked] = data.runs;
    const size = data.config.subject_size;
    const peak = peaked.bins.reduce((best, bin) => bin.observed > best.observed ? bin : best);
    const flatCount = flat.bins.find((bin) => bin.level === peak.level).observed;
    const scenes = [
        [0, 'One hospital. Two staff profiles.', 'Change one training-level sketch to generate a different access-control test dataset.'],
        [9, 'Describe the hospital.', 'The JSON sets the entity and rule counts. The ZIP contains one distribution sketch per attribute.'],
        [22, 'See what MuSimA read.', peaked.detail + ' The progress sequence is shortened for the replay.'],
        [34, 'Check the generated distribution.', `At training level ${peak.level}, the peaked run produced ${peak.observed} of ${size} staff. The flat run produced ${flatCount}. The line shows expected counts and the bars show the generated counts.`],
        [49, 'Sample a permit/deny log.', 'MuSimA evaluates the generated policy to build the access matrix. Generate Logs samples that matrix. This example requests 1,000 rows with 20% permits.'],
        [61, 'Try the same experiment.', 'Download both inputs and outputs to reproduce the comparison, or open the live workspace to generate your own dataset.'],
    ];

    function svg(name, attributes, content = '') {
        const node = document.createElementNS('http://www.w3.org/2000/svg', name);
        for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
        node.textContent = content;
        return node;
    }

    function renderCharts(container, showExpected) {
        const maxCount = Math.max(...data.runs.flatMap((run) => run.bins.flatMap(({expected, observed}) => [expected, observed])));
        const axisMax = Math.ceil(maxCount / 20) * 20;
        const y = (count) => 170 - count / axisMax * 125;
        const step = 322 / flat.bins.length;
        const x = (index) => 42 + step * (index + 0.5);

        for (const run of data.runs) {
            const figure = document.createElement('figure');
            const caption = document.createElement('figcaption');
            caption.textContent = `${run.label} · ${run.family}`;
            const counts = run.bins.map(({level, expected, observed}) =>
                `Level ${level}: ${observed} observed${showExpected ? `, ${expected} expected` : ''}`);
            const chart = svg('svg', {
                viewBox: '0 0 390 226',
                role: 'img',
                'aria-label': `${run.label} staff counts by training level. ${counts.join('. ')}.`,
            });
            chart.append(svg('text', {x: 42, y: 25}, 'Staff count'));
            for (let index = 0; index <= 4; index++) {
                const count = axisMax * index / 4;
                chart.append(svg('line', {x1: 42, x2: 364, y1: y(count), y2: y(count)}));
                chart.append(svg('text', {x: 34, y: y(count) + 4, 'text-anchor': 'end'}, count));
            }
            run.bins.forEach(({level, observed}, index) => {
                chart.append(svg('rect', {x: x(index) - step * 0.32, y: y(observed), width: step * 0.64, height: 170 - y(observed), rx: 2}));
                chart.append(svg('text', {x: x(index), y: 190, 'text-anchor': 'middle'}, level));
            });
            if (showExpected) {
                const points = run.bins.map(({expected}, index) => `${x(index)},${y(expected)}`).join(' ');
                chart.append(svg('polyline', {points}));
            }
            chart.append(svg('text', {x: 203, y: 216, 'text-anchor': 'middle'}, 'Staff training level'));
            figure.append(caption, chart);
            container.append(figure);
        }
    }

    renderCharts($('demo-overview'), false);
    renderCharts($('demo-comparison'), true);
    const staticAsset = new URL('demo/', document.currentScript.src);
    $('demo-fit-image').src = new URL(peaked.fit, staticAsset);
    $('demo-permit').textContent = peaked.rules.permit;
    $('demo-deny').textContent = peaked.rules.deny;
    for (const row of peaked.logs) {
        const tr = $('demo-log-rows').insertRow();
        for (const value of row) tr.insertCell().textContent = value;
    }
    $('num-logs').value = 1000;
    $('allow-percentage').value = 20;
    $('log-gen-empty-state').hidden = true;
    for (const id of ['log-gen-tab-btn', 'num-logs', 'allow-percentage', 'generate-logs-btn']) $(id).disabled = false;
    $('log-gen-tab-btn').removeAttribute('title');

    const duration = Number(timeline.max);
    let elapsed = 0;
    let origin = performance.now();
    let playing = !reducedMotion;
    let frame;
    let active = -1;
    let phase = '';

    function showScene(index) {
        const tab = ['', 'multimodal', 'multimodal', 'multimodal', 'log-gen', ''][index];
        for (const button of tabs) button.classList.toggle('active', button.dataset.tab === tab);
        $('multimodal-tab').classList.toggle('active', index === 1);
        $('log-gen-tab').classList.toggle('active', index === 4);
        $('demo-overview').hidden = index !== 0 && index !== 5;
        $('demo-fit').hidden = index !== 2;
        $('progress-container').style.display = index === 2 ? 'block' : 'none';
        $('download-container').style.display = index === 3 ? 'block' : 'none';
        $('demo-actions').hidden = index !== 5;
        $('demo-number').textContent = String(index + 1).padStart(2, '0');
        $('demo-title').textContent = scenes[index][1];
        $('demo-caption').textContent = scenes[index][2];
        for (const button of chapters) button.removeAttribute('aria-current');
        chapters[index].setAttribute('aria-current', 'step');
        phase = '';
        scroller.scrollTop = 0;
    }

    function showPhase(index, seconds) {
        if (index === 1) {
            const next = seconds < 4 ? 'config' : seconds < 8 ? 'zip' : 'submit';
            if (phase === next) return;
            phase = next;
            const form = $('multimodal-form');
            form.dataset.demoPhase = next;
            $('demo-config').textContent = JSON.stringify(data.config, null, 2);
            $('zip-name').textContent = next === 'config' ? 'Choose a ZIP file' : peaked.zip;
            if (!reducedMotion) {
                const target = {config: 'min-json-editor', zip: 'zip-wrapper', submit: 'submit-multimodal-btn'}[next];
                const distance = $(target).getBoundingClientRect().top - scroller.getBoundingClientRect().top;
                scroller.scrollTo({top: scroller.scrollTop + distance - 18, behavior: 'smooth'});
            }
        }
        if (index === 2) {
            const next = seconds < 2 ? 0 : seconds < 4 ? 1 : seconds < 6 ? 2 : 3;
            if (phase === next) return;
            phase = next;
            const steps = $('progress-container').querySelectorAll('.step');
            steps.forEach((step, position) => {
                step.classList.toggle('active', position === next);
                step.classList.toggle('completed', position < next);
            });
            $('demo-fit-image').hidden = next < 3;
        }
    }

    function display() {
        const index = scenes.findLastIndex(([start]) => start <= elapsed);
        if (active !== index) {
            active = index;
            showScene(index);
        }
        showPhase(index, elapsed - scenes[index][0]);
        timeline.value = elapsed;
    }

    function tick(now) {
        elapsed = Math.max(0, (now - origin) / 1000) % duration;
        display();
        frame = requestAnimationFrame(tick);
    }

    function seek(time) {
        elapsed = time;
        origin = performance.now() - elapsed * 1000;
        display();
    }

    function setPlaying(next) {
        cancelAnimationFrame(frame);
        playing = next;
        toggle.textContent = playing ? 'Pause' : 'Resume';
        if (playing) {
            origin = performance.now() - elapsed * 1000;
            frame = requestAnimationFrame(tick);
        }
    }

    toggle.addEventListener('click', () => setPlaying(!playing));
    $('demo-restart').addEventListener('click', () => {
        seek(0);
        setPlaying(!reducedMotion);
    });
    chapters.forEach((button, index) => button.addEventListener('click', () => {
        const previewOffset = playing ? 0 : [0, 8, 6, 0, 0, 0][index];
        seek(scenes[index][0] + previewOffset);
    }));
    for (const link of document.querySelectorAll('.demo-mode a[download]')) {
        link.addEventListener('focus', () => setPlaying(false));
    }
    display();
    setPlaying(playing);
})();
