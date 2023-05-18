import tonic
import torch, torchvision
ncaltech101 = tonic.datasets.NCALTECH101("/Users/filippocasari/GDLProject/data/extract")

events, label = ncaltech101[500]
print(label)
#ani = tonic.utils.plot_event_grid(events)
print(events, label)
tonic.utils.plot_event_grid(events, axis_array=(3,3), plot_frame_number=True)
sensor_size = ncaltech101.sensor_size
transform = tonic.transforms.ToVoxelGrid(
    sensor_size= (200, 200, 2),
    n_time_bins=70,
    
)

frames = transform(events)
ani = tonic.utils.plot_animation(frames)
