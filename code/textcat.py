#!/usr/bin/env python3
"""
Computes the total log probability of the sequences of tokens in each file,
according to a given smoothed trigram model.  
"""
import argparse
import logging
import math
from pathlib import Path
import torch

from probs import Wordtype, LanguageModel, num_tokens, read_trigrams

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_1",
        type=Path,
        help="path to the first trained model",
    )
    parser.add_argument(
        "model_2",
        type=Path,
        help="path to the second trained model",
    )
    parser.add_argument(
        "prob_1",
        type=float,
        help="prior probability that a file is from the first model",
    )
    parser.add_argument(
        "test_files",
        type=Path,
        nargs="*"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=['cpu','cuda','mps'],
        help="device to use for PyTorch (cpu or cuda, or mps if you are on a mac)"
    )

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   dest="logging_level", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()


def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    log_prob = 0.0

    x: Wordtype; y: Wordtype; z: Wordtype    # type annotation for loop variables below
    for (x, y, z) in read_trigrams(file, lm.vocab):
        log_prob += lm.log_prob(x, y, z)  # log p(z | xy)

        # If the factor p(z | xy) = 0, then it will drive our cumulative file 
        # probability to 0 and our cumulative log_prob to -infinity.  In 
        # this case we can stop early, since the file probability will stay 
        # at 0 regardless of the remaining tokens.
        if log_prob == -math.inf: break 

        # Why did we bother stopping early?  It could occasionally
        # give a tiny speedup, but there is a more subtle reason -- it
        # avoids a ZeroDivisionError exception in the unsmoothed case.
        # If xyz has never been seen, then perhaps yz hasn't either,
        # in which case p(next token | yz) will be 0/0 if unsmoothed.
        # We can avoid having Python attempt 0/0 by stopping early.
        # (Conceptually, 0/0 is an indeterminate quantity that could
        # have any value, and clearly its value doesn't matter here
        # since we'd just be multiplying it by 0.)

    return log_prob


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    # Check if prob_1, the prior probability that a file is from the first model,
    # is between 0 and 1, inclusive.
    if args.prob_1 < 0 or 1 < args.prob_1:
        logging.critical("The prior probability that a file is from the first model must be "
                         "between 0 and 1, inclusive.")
        exit(1)

    # Specify hardware device where all tensors should be computed and
    # stored.  This will give errors unless you have such a device
    # (e.g., 'gpu' will work in a Kaggle Notebook where you have
    # turned on GPU acceleration).
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                logging.critical("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled.")
            else:
                logging.critical("MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine.")
            exit(1)
    torch.set_default_device(args.device)
        
    log.info("Testing...")
    lm_1 = LanguageModel.load(args.model_1, device=args.device)
    lm_2 = LanguageModel.load(args.model_2, device=args.device)

    # Check if the two models use the same vocabulary.
    if lm_1.vocab != lm_2.vocab:
        logging.critical("The two models must use the same vocabulary file.")
        exit(1)

    log.info("Per-file categorization:")

    model_1_n = 0
    model_2_n = 0

    log_prob_1 = -math.inf if args.prob_1 == 0 else math.log(args.prob_1)
    log_prob_2 = -math.inf if args.prob_1 == 1 else math.log(1 - args.prob_1)

    for file in args.test_files:
        log_prob_text_and_1:float = log_prob_1 + file_log_prob(file, lm_1)
        log_prob_text_and_2:float = log_prob_2 + file_log_prob(file, lm_2)

        if log_prob_text_and_1 > log_prob_text_and_2:
            model = args.model_1
            model_1_n += 1
        else:
            model = args.model_2
            model_2_n += 1

        print(f"{model}\t{file}")

    total_n = model_1_n + model_2_n
    print(f"{model_1_n} files were more probably from {args.model_1} ({model_1_n / total_n * 100:.2f}%)")
    print(f"{model_2_n} files were more probably from {args.model_2} ({model_2_n / total_n * 100:.2f}%)")


if __name__ == "__main__":
    main()

