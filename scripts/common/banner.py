# fmt: off
BANNERS = {
    # vanilla DreamerV3 baseline under non-stationary dynamics
    'continual_baseline': [
        r"---  ___           _   _ _          ---",
        r"--- | _ ) __ _ ___| |_| (_)_ _  ___ ---",
        r"--- | _ \/ _` (_-<  _| | | ' \/ -_) ---",
        r"--- |___/\__,_/__/\__|_|_|_||_\___| ---",
        r"---         Continual Baseline      ---",
    ],

    # random agent data collection (no training)
    'random_agent': [
        r"---  ___                           __   ______ ---",
        r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
        r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
        r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
        r"---               Random Dreamer               ---",
    ],

    # fixed-dynamics DreamerV3 + periodic HDF5 offline data collection
    'per_dy_dreamer': [
        r"---  ___                           __   ______ ---",
        r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
        r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
        r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---",
        r"---            Per Dynamics Dreamer            ---",
    ],

    # Inception pretraining on offline HDF5 data
    'inception': [
        r"---  ___                    _   _           ---",
        r"--- |_ _|_ _  __ ___ _ __ | |_(_) ___ _ _   ---",
        r"---  | || ' \/ _/ -_) '_ \|  _| |/ _ \ ' \  ---",
        r"--- |___|_||_\__\___| .__/ \__|_|\___/_||_| ---",
        r"---                 |_|                     ---",
        r"---                Inception                ---",
    ],

    # DreamerV3 + Inception under non-stationary dynamics
    'lucid_dreamer': [
        r"---  _           _    _  ___                           __   ______  ---",
        r"--- | |   _  _ __(_)__| ||   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---",
        r"--- | |__| || / _| / _` || |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---",
        r"--- |____\_, \__|_\__,_||___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/  ---",
        r"---       |__/                                                      ---",
        r"---                          Lucid Dreamer                          ---",
    ],

}
# fmt: on


def print_banner(task: str):
    import elements
    lines = BANNERS.get(task)
    if lines:
        for line in lines:
            elements.print(line)
