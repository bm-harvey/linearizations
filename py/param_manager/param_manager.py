import os


class ParamManager:
    def __init__(self, dir: str = None):
        # global
        self.x_col: str = "x"
        self.y_col: str = "y"
        # ridge finding
        self.sigma: float = 1.0
        self.threshold: float = 5.0e-4
        self.num_bins_x: int = 1000
        self.num_bins_y: int = 1000
        # lowess smoothing
        self.bandwidth: float = 0.1
        self.polynomial_order: int = 2
        # linearize
        self.num_extra_lines: int = 0

        file = os.path.join(dir, "params.dat")
        self.read_params_from_file(file)

    def read_params_from_file(self, file: str):
        for line in open(file, "r"):
            line = line.strip()
            if line.strip() == "":
                continue
            if line[0] == "#":
                continue
            words = line.split(":")

            param_name = words[0].strip()
            param_value = words[1].strip()

            match param_name:
                case "x_col":
                    self.x_col = param_value
                case "y_col":
                    self.y_col = param_value
                case "sigma":
                    self.sigma = float(param_value)
                case "threshold":
                    self.threshold = float(param_value)
                case "bins_x":
                    self.num_bins_x = int(param_value)
                case "bins_y":
                    self.num_bins_y = int(param_value)
                case "bandwidth":
                    self.bandwidth = float(param_value)
                case "polynomial_order":
                    self.polynomial_order = int(param_value)
                case "num_extra_lines":
                    self.num_extra_lines = int(param_value)
                case _:
                    print(f"Parameter {param_name} not recognized or used.")

        # self.bandwidth: float = 0.1
        # self.polynomial_order: int = int

        # file = os.path.join(dir, "params.dat")
        # self.read_params_from_file(file)


if __name__ == "__main__":
    ParamManager("data/faust_det_60")
