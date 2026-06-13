import torch

from mod_ani.config import quick_test_config
from mod_ani.descriptors import HydrogenLikeRadial, make_hydrogen_like_aev
from mod_ani.local_torchani import torchani_source_path, use_local_torchani
from mod_ani.models import build_model
from mod_ani.training import filter_energy_outliers, move_batch


def test_hydrogen_like_radial_shape_and_grad():
    radial = HydrogenLikeRadial.low_quantum_numbers()
    distances = torch.tensor([0.8, 1.2, 2.0], requires_grad=True)

    features = radial(distances)

    assert features.shape == (3, radial.num_feats)
    assert torch.isfinite(features).all()
    features.sum().backward()
    assert distances.grad is not None
    assert torch.isfinite(distances.grad).all()


def test_hydrogen_like_radial_includes_3d_feature():
    radial = HydrogenLikeRadial.low_quantum_numbers()

    assert radial.num_feats == 6
    assert (
        float(radial.principal.flatten()[-1]),
        float(radial.angular_momentum.flatten()[-1]),
    ) == (3.0, 2.0)


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


def test_move_batch_casts_float64_before_device_transfer():
    batch = {
        "species": torch.tensor([[0, 1]], dtype=torch.long),
        "coordinates": torch.zeros((1, 2, 3), dtype=torch.float64),
        "energies": torch.zeros(1, dtype=torch.float64),
    }

    moved = move_batch(batch, torch.device("cpu"), torch.float32)

    assert moved["species"].dtype == torch.long
    assert moved["coordinates"].dtype == torch.float32
    assert moved["energies"].dtype == torch.float32


def test_filter_energy_outliers_drops_placeholder_scale_targets():
    batch = {
        "species": torch.tensor([[0, 1], [0, 1], [0, 1]], dtype=torch.long),
        "coordinates": torch.zeros((3, 2, 3), dtype=torch.float32),
        "energies": torch.tensor([-40.0, 2.0e10, -41.0], dtype=torch.float32),
    }

    filtered = filter_energy_outliers(batch, max_abs_energy_hartree=1.0e5)

    assert filtered is not None
    assert filtered["species"].shape[0] == 2
    assert filtered["coordinates"].shape[0] == 2
    assert filtered["energies"].tolist() == [-40.0, -41.0]
