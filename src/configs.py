configs = {
    "MobiAct": {
        "user_split": False,
        "window_size": 60,
        "window_step": 20,
        "epochs": 50,
        "two_class": False,
        "two_class_classification": False,
        "extract_fall": False,
        "frequancy": "20ms",
        "normlize": True,
        "balance": True,
        "balance_classification": True,
    },
    "SiSFall": {
        "user_split": False,
        "window_size": 40,
        "window_step": 10,
        "epochs": 50,
        "two_class": True,
        "two_class_classification": True,
        "extract_fall": True,
        "frequancy": "50ms",
        "normlize": True,
        "balance": False,
    }
}