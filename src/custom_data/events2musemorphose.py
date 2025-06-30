import glob
import pickle
from pathlib import Path

from tqdm.auto import tqdm

# events_path_default = "./ailab17k_from-scratch_remi"
# pickle_path = "../pickles/custom_test_pieces.pkl"

events_path_default = "./symphonynet_classical_remi/events"
musemorphose_path_default = "./symphonynet_classical_remi"


def pickle_load(f):
    return pickle.load(open(f, "rb"))


def pickle_dump(obj, f):
    pickle.dump(obj, open(f, "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def main(
    events_path: str = events_path_default,
    musemorphose_path: str = musemorphose_path_default,
    tokenization: str = "remi",
    VERBOSE: bool = True,
):
    """Converts event files into musemorphose files

    Parameters
    ----------
    events_path : str, optional
        Input path, by default events_path_default
    musemorphose_path : str, optional
        Output path, by default musemorphose_path_default
    tokenization : str, optional
        Tokenization strategy, by default "remi"
    VERBOSE : bool, optional
        Show output, by default True
    """
    test_pieces = []
    files = list(Path(events_path).glob("**/*.pkl"))

    if VERBOSE:
        print("num files:", len(files))
    for orig_file in tqdm(files, total=len(files), leave=False):
        out_file = str(orig_file).replace(
            str(Path(events_path)), str(Path(musemorphose_path))
        )
        events = pickle_load(orig_file)

        # REMI tokenization
        if tokenization == "remi":
            for event in events:
                if event["name"] == "Note_Velocity":
                    event["value"] = min(max(40, event["value"]), 80)
            bar_idx = []
            for idx, event in enumerate(events):
                if event["name"] == "Bar":
                    bar_idx.append(idx)

        # CPWord tokenization
        elif tokenization == "cpword":
            for event in events:
                if event["family"] == "Note":
                    event["velocity"] = min(max(40, event["velocity"]), 80)
            bar_idx = []
            for idx, event in enumerate(events):
                if event["family"] == "Bar":
                    bar_idx.append(idx)

        result = (bar_idx, events)
        test_pieces.append(out_file.split("/")[-1])
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        pickle_dump(result, out_file)


if __name__ == "__main__":
    main()
