import torch


def grid_box_bias(num_patches_height: int, num_patches_width: int) -> torch.Tensor:
    x_coordinate = torch.arange(1, num_patches_width + 1, dtype=torch.float32)
    y_coordinate = torch.arange(1, num_patches_height + 1, dtype=torch.float32)
    xx, yy = torch.meshgrid(x_coordinate, y_coordinate, indexing="xy")

    box_coordinates = torch.stack(tensors=[xx, yy], dim=-1)
    box_coordinates[..., 0] /= num_patches_width
    box_coordinates[..., 1] /= num_patches_height
    box_coordinates = box_coordinates.reshape(-1, 2).clamp(0.0, 1.0)

    # box anchors - reversed sigmoid: log(x) - log(1 - x)
    box_coordinates_bias = torch.log(input=box_coordinates + 1e-4) - torch.log1p(input=-box_coordinates + 1e-4)

    box_size = torch.ones_like(input=box_coordinates_bias)
    box_size[..., 0] /= num_patches_height
    box_size[..., 1] /= num_patches_width
    box_size_bias = torch.log(input=box_size + 1e-4) - torch.log1p(input=-box_size + 1e-4)

    return torch.cat([box_coordinates_bias, box_size_bias], dim=-1)