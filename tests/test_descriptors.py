import torch

from mod_ani.descriptors import HydrogenLikeRadial, make_hydrogen_like_aev


def test_hydrogen_like_radial_shape_and_grad():
    radial = HydrogenLikeRadial.low_quantum_numbers()
    distances = torch.tensor([0.8, 1.2, 2.0], requires_grad=True)

    features = radial(distances)

    assert features.shape == (3, radial.num_feats)
    assert torch.isfinite(features).all()
    features.sum().backward()
    assert distances.grad is not None
    assert torch.isfinite(distances.grad).all()


def test_hydrogen_like_aev_shape():
    aev = make_hydrogen_like_aev(num_species=4)
    species = torch.tensor([[0, 1, 1]])
    coordinates = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.1], [1.0, 0.0, 0.0]]]
    )

    features = aev(species, coordinates)

    assert features.shape == (1, 3, aev.out_dim)
    assert aev.radial_len == 4 * aev.radial.num_feats
