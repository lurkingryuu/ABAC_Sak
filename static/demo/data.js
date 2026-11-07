window.MUSIMA_DEMO = {
  "config": {
    "subject_size": 200,
    "object_size": 30,
    "environment_size": 3,
    "permit_rules_count": 5,
    "deny_rules_count": 3
  },
  "runs": [
    {
      "label": "Flat",
      "family": "Uniform",
      "zip": "flat-input.zip",
      "fit": null,
      "detail": "Gemini fitted a Uniform distribution across the twelve training levels.",
      "bins": [
        {
          "level": 1,
          "expected": 16.67,
          "observed": 12
        },
        {
          "level": 2,
          "expected": 16.67,
          "observed": 13
        },
        {
          "level": 3,
          "expected": 16.67,
          "observed": 16
        },
        {
          "level": 4,
          "expected": 16.67,
          "observed": 21
        },
        {
          "level": 5,
          "expected": 16.67,
          "observed": 22
        },
        {
          "level": 6,
          "expected": 16.67,
          "observed": 19
        },
        {
          "level": 7,
          "expected": 16.67,
          "observed": 15
        },
        {
          "level": 8,
          "expected": 16.67,
          "observed": 12
        },
        {
          "level": 9,
          "expected": 16.67,
          "observed": 20
        },
        {
          "level": 10,
          "expected": 16.67,
          "observed": 14
        },
        {
          "level": 11,
          "expected": 16.67,
          "observed": 19
        },
        {
          "level": 12,
          "expected": 16.67,
          "observed": 17
        }
      ],
      "rules": {
        "permit": "SA_1 = SA_1_12, OA_1 = OA_1_1, EA_1 = EA_1_1",
        "deny": "SA_1 = SA_1_9, OA_1 = OA_1_1, EA_1 = EA_1_2"
      },
      "logs": [
        [
          "S_113",
          "O_12",
          "E_3",
          "permit"
        ],
        [
          "S_154",
          "O_12",
          "E_1",
          "deny"
        ],
        [
          "S_142",
          "O_8",
          "E_3",
          "deny"
        ]
      ]
    },
    {
      "label": "Peaked",
      "family": "Normal",
      "zip": "peaked-input.zip",
      "fit": "peaked-fit.png",
      "detail": "Gemini fitted a Normal curve centered at level 9.",
      "bins": [
        {
          "level": 1,
          "expected": 0.12,
          "observed": 0
        },
        {
          "level": 2,
          "expected": 0.42,
          "observed": 0
        },
        {
          "level": 3,
          "expected": 1.28,
          "observed": 1
        },
        {
          "level": 4,
          "expected": 3.29,
          "observed": 2
        },
        {
          "level": 5,
          "expected": 7.24,
          "observed": 4
        },
        {
          "level": 6,
          "expected": 13.62,
          "observed": 12
        },
        {
          "level": 7,
          "expected": 21.88,
          "observed": 21
        },
        {
          "level": 8,
          "expected": 30.0,
          "observed": 35
        },
        {
          "level": 9,
          "expected": 35.13,
          "observed": 41
        },
        {
          "level": 10,
          "expected": 35.13,
          "observed": 32
        },
        {
          "level": 11,
          "expected": 30.0,
          "observed": 30
        },
        {
          "level": 12,
          "expected": 21.88,
          "observed": 22
        }
      ],
      "rules": {
        "permit": "SA_1 = SA_1_12, OA_1 = OA_1_1, EA_1 = EA_1_1",
        "deny": "SA_1 = SA_1_11, OA_1 = OA_1_1, EA_1 = EA_1_2"
      },
      "logs": [
        [
          "S_16",
          "O_9",
          "E_2",
          "permit"
        ],
        [
          "S_125",
          "O_29",
          "E_3",
          "deny"
        ],
        [
          "S_88",
          "O_7",
          "E_1",
          "deny"
        ]
      ]
    }
  ]
};
