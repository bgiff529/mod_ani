import torch

from mod_ani.config import quick_test_config
from mod_ani.descriptors import HydrogenLikeRadial, make_hydrogen_like_aev
from mod_ani.local_torchani import torchani_source_path, use_local_torchani
from mod_ani.models import build_model


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


def test_torchani_imports_from_local_vendor_checkout():
    use_local_torchani()
    import torchani

    assert str(torchani_source_path()) in str(torchani.__file__)
    assert torchani_source_path().exists()


def test_electron_radial_model_uses_hydrogen_like_radial():
    model = build_model(quick_test_config(model_kind="electron_radial"))

    assert isinstance(model.aev_computer.radial, HydrogenLikeRadial)
