from aegnn.visualize.data import *
from aegnn.datasets import *
import aegnn
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib
# Run PYTHONPATH=$(pwd) AEGNN_DATA_DIR=$(pwd) AEGNN_LOG_DIR=$(pwd)/data/log python plotting_ncars.py --dataset "ncars"
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")

    parser = aegnn.datasets.EventDataModule.add_argparse_args(parser)
    return parser.parse_args()



def read_sequence_files(path):
    sequences = []
    counter_no_cars = 0
    counter_cars = 0
    for root, dirs, files in os.walk(path):
        for directory in dirs:
            if directory.startswith("sequence_"):
                sequence_path = os.path.join(root, directory)
                for file in os.listdir(sequence_path):
                    if(file == "is_car.txt"):
                        file_path = os.path.join(sequence_path, file)

                        with open(file_path, "r") as f:
                            is_car = int(f.read().strip())
                            if(is_car == 1):
                                counter_cars += 1
                                sequences.append("car")
                            else:
                                counter_no_cars += 1
                                sequences.append("no car")
                            #sequences.append()
    return sequences, counter_cars, counter_no_cars

def extract_positions(data: torch_geometric.data.Data):

    pos_x = data.pos[:, 0]
    pos_y = data.pos[:, 1]
    edges = []
    if (edge_index := getattr(data, "edge_index")) is not None:
        for edge in tqdm.tqdm(edge_index.T):
            pos_edge = data.pos[[edge[0], edge[1]], :]
            edges.append([pos_edge[:, 0],pos_edge[:, 1] ])
            

    return edges, pos_x, pos_y

def main(args):

    choise = int(input("Insert:\n(1) For visualizing the graphs\n(2) For visualizing the histograms\n"))
    if(choise != 1 and choise!=2):
        print("you must choose between 1 and 2")
        return -1
    dm = aegnn.datasets.by_name(args.dataset).from_argparse_args(args)
    dm.setup()
    print(dm.train_dataset[0])

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    path_labels = "ncars/training"
    
    labels, cars, no_cars = read_sequence_files(path_labels)
    
    #print(labels)
    print(f"cars: {cars}, no cars: {no_cars}")
    
    list_train = [dm.train_dataset[i] for i in range(100)]
    
    
    for event, label  in zip(list_train, labels[:100]):
        ax.cla()
        #edges = event[0]
        #pos_edges_0, pos_edges_1 = edges[0], edges[1]
        #
        #print(event.edge_index[0, :10])
        #print(event.edge_index[1, :10])
        if(choise == 1):
            ax = graph(event, ax)
        else:
            ax = event_histogram(event,title= label,  ax=ax)
        plt.draw()
        plt.pause(2)
    #ax.cla()
    #ax = event_histogram(dm.train_dataset[0])
    #plt.
    #plt.show()
    #print(type(dm))
    

    


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)