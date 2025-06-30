class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_events(events):
    for ev in events:
        if isinstance(ev, dict) and "name" in ev.keys() and "value" in ev.keys():
            col = ""
            before = ""
            if ev["name"] == "Bar":
                col = bcolors.OKBLUE
                before = "\n"
            if ev["name"] == "Beat":
                col = bcolors.OKCYAN
            if ev["name"] == "Track":
                col = bcolors.OKGREEN
            if "Melody" in ev["name"]:
                col = bcolors.WARNING
            print(f"{before}{col}{ev}{bcolors.ENDC}")
        else:
            print(ev)
