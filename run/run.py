from scripts.experiment import Experiment


def main():
    ex = Experiment()
    y_pred = ex.run(nrows=None)
    return y_pred


if __name__ == "__main__":
    main()
