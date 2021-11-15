```
  while True:
    image_idx = int(input("Enter image idx"))
    world_pt_idx = [ idx for _, idx in images[registered_images[image_idx]].p3D_idxs.items()]
    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection='3d')
    Plot3DPoints(points3D[world_pt_idx], ax3d)
    PlotCameras(images, [registered_images[image_idx],], ax3d)

    # Delay termination of the program until the figures are closed
    # Otherwise all figure windows will be killed with the program
    plt.show(block=True)
```