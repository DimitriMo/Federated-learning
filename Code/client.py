import argparse
import flwr as fl
import multiprocessing as mp
from flower_helpers import train, test
import matplotlib.pyplot as plt

"""
If you get an error like: “failed to connect to all addresses” “grpc_status”:14 
Then uncomment the lines bellow:
"""
import os
if os.environ.get("https_proxy"):
    del os.environ["https_proxy"]
if os.environ.get("http_proxy"):
    del os.environ["http_proxy"]

def main():
    """Get all args necessary for dp"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=int, default=0, help="Client number for dataset share")
    parser.add_argument("-r", type=int, default=4, help="Number of rounds for the federated training")
    parser.add_argument("-nbc",type=int,default=2,help="Number of clients to keep track of dataset share",)
    parser.add_argument("-vb", type=int, default=256, help="Virtual batch size")
    parser.add_argument("-b", type=int, default=256, help="Batch size")
    parser.add_argument("-lr", type=float, default=0.0001, help="Learning rate for the optimizer")
    parser.add_argument("-nm", type=float, default=1.2, help="Noise multiplier for Private Engine.")
    parser.add_argument("-mgn", nargs= '+', type=float, default=1.0, help="Max grad norm for Private Engine.")
    parser.add_argument("-eps",type=float,default=0,help="Target epsilon for the privacy budget.",)
    
    args = parser.parse_args()
    client_share = int(args.c)
    nbc = int(args.nbc)
    vbatch_size = int(args.vb)
    batch_size = int(args.b)
    lr = float(args.lr)
    nm = float(args.nm)
    mgn = float(args.mgn[0])
    eps = float(args.eps)
    rounds = int(args.r)

    
    """Create model, load data, define Flower client, start Flower client."""

    # Set the start method for multiprocessing in case Python version is under 3.8.1
    mp.set_start_method("spawn")

    # Flower client
    class CifarDpClient(fl.client.NumPyClient):
        def __init__(
            self, 
            client_share: int,
            nbc: int,
            vbatch_size: int,
            batch_size: int,
            rounds: int,
            lr: float,
            eps: float,
            nm: float,
            mgn: float,
            ):
                        
            """Differentially private implementation of a Cifar client.

            Parameters
            ----------
            client_share : int
                Share of client to determine it's dataset.
            nbc : int
                Number of clients to load the correct dataset folder.
            vbatch_size : int
                Virtual batch size.
            batch_size : int
                Batch size.
            lr : float
                Learning rate.
            eps : float
                Target epsilon.
            nm : float
                Noise multiplier.
            mgn : float
                Maximum gradient norm.
            """
            self.client_share = client_share
            self.nbc = nbc
            self.vbatch_size = vbatch_size
            self.batch_size = batch_size
            self.lr = lr
            self.eps = eps
            self.nm = nm
            self.mgn = mgn
            self.avg_gradients = {"conv1": [], "conv2": [], "fc1": [], "fc2": [], "fc3": []}
            self.round_number = 0
            self.rounds = rounds
            self.track_epsilon = []
            self.track_accuracy = []
            self.parameters = None
            self.state_dict = None


        def get_parameters(self):
            return self.parameters

        def set_parameters(self, parameters):
            self.parameters = parameters

        def fit(self, parameters, config):
            self.round_number += 1
            self.set_parameters(parameters)
            # Prepare multiprocess
            manager = mp.Manager()
            # We receive the results through a shared dictionary
            return_dict = manager.dict()
            p = mp.Process(
                target=train,
                args=(
                    parameters,
                    return_dict,
                    config,
                    self.client_share,
                    self.nbc,
                    self.vbatch_size,
                    self.batch_size,
                    self.lr,
                    self.nm,
                    self.mgn,
                    self.state_dict,
                    self.avg_gradients,
                    self.track_epsilon,
                    self.track_accuracy,
                    )
                )
            # Start the process
            p.start()
            # Wait for it to end
            p.join()
            # Close it
            try:
                p.close()
            except ValueError as e:
                print(f"Couldn't close the training process: {e}")

            # Get the return values
            new_parameters = return_dict["parameters"]
            data_size = return_dict["data_size"]
            # Store updated state dict
            self.state_dict = return_dict["state_dict"]
            self.avg_gradients = return_dict["avg_gradients"]
            self.track_epsilon = return_dict["track_epsilon"]
            self.track_accuracy = return_dict["track_accuracy"]
            
            
            if self.round_number == self.rounds:
                
                #####PLOT#####
                data = []
                for grad in self.avg_gradients.values():
                    data.append(grad)
                layer_names = ['Conv1', 'Conv2', 'Lin1', 'Lin2', 'Lin3']
                

                plt.boxplot(data, labels=layer_names)

                plt.xlabel('Layers')
                plt.ylabel('Avg_gradients')

                plt.show()
                
                avg_epsi = sum(self.track_epsilon)/self.round_number
                avg_accuracy = sum(self.track_accuracy)/self.round_number
                
                print(f"AVERAGE EPSILON: {avg_epsi}")
                
                save_path = "per_layer.txt"

                with open(save_path, "r+") as f:
                    lines = f.readlines()
                    found_empty_line = False

                    for i, line in enumerate(lines):
                        if not line.strip():
                            found_empty_line = True
                            lines[i] = f"{i + 1}: {avg_epsi}, with the accuracy: {avg_accuracy}\n"
                            break

                    if not found_empty_line:
                        lines.append(f"{len(lines) + 1}: epsilon: {avg_epsi}, with the accuracy: {avg_accuracy}\n")

                    f.seek(0)
                    f.writelines(lines)
                                                                                                                                
                #####PLOT#####
                
                
                
                                
            # Check if target epsilon value is respected
            accept = True
            # Leave +0.3 margin to accomodate with opacus imprecisions
            if return_dict["eps"] > self.eps + 0.3:
                # refuse the client new parameters
                accept = False
                print(
                    f"Epsilon over target value ({self.eps}), disconnecting client."
                )
                # Override new parameters with previous ones
                new_parameters = parameters
                print()
            # Init metrics dict
            metrics = {
                "epsilon": return_dict["eps"],
                "alpha": return_dict["alpha"],
                "accept": accept,
            }
            # Del everything related to multiprocessing
            del (manager, return_dict, p)
            return new_parameters, data_size, metrics

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            # Prepare multiprocess
            manager = mp.Manager()
            # We receive the results through a shared dictionary
            return_dict = manager.dict()
            # Create the process
            p = mp.Process(target=test, args=(
                parameters,
                return_dict,
                self.client_share,
                self.nbc,
                self.batch_size
                ))
            # Start the process
            p.start()
            # Wait for it to end
            p.join()
            # Close it
            try:
                p.close()
            except ValueError as e:
                print(f"Coudln't close the evaluating process: {e}")
                                                
                
            # Get the return values
            loss = return_dict["loss"]
            accuracy = return_dict["accuracy"]
            data_size = return_dict["data_size"]
            # Del everything related to multiprocessing
            del (manager, return_dict, p)
                    
            return float(loss), data_size, {"accuracy": float(accuracy)}

    # Start client
    fl.client.start_numpy_client("[::]:8080", client=CifarDpClient(
        client_share,
        nbc,
        vbatch_size,
        batch_size,
        rounds,
        lr,
        eps,
        nm,
        mgn,
        ))


if __name__ == "__main__":
    main()
