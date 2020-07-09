"""Tests for funcitons that identify system characteristics."""
import pytest
import pandas as pd
from pvlib import location, pvsystem, tracking, modelchain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from pvanalytics import system


@pytest.fixture(scope='module')
def summer_times():
    """One hour time stamps from May 1 through September 30, 2020 in GMT+7"""
    return pd.date_range(
        start='2020-5-1',
        end='2020-10-1',
        freq='H',
        closed='left',
        tz='Etc/GMT+7'
    )


@pytest.fixture(scope='module')
def albuquerque():
    """pvlib Location for Albuquerque, NM."""
    return location.Location(
        35.0844,
        -106.6504,
        name='Albuquerque',
        altitude=1500,
        tx='Etc/GMT+7'
    )


@pytest.fixture
def system_parameters():
    """System parameters for generating simulated power data."""
    sandia_modules = pvsystem.retrieve_sam('SandiaMod')
    sapm_inverters = pvsystem.retrieve_sam('cecinverter')
    module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
    inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
    temperature_model_parameters = (
        TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    )
    return {
        'module_parameters': module,
        'inverter_parameters': inverter,
        'temperature_model_parameters': temperature_model_parameters
    }


@pytest.fixture(scope='module')
def summer_clearsky(summer_times, albuquerque):
    """Clearsky irradiance for `sumer_times` in Albuquerque, NM."""
    return albuquerque.get_clearsky(summer_times, model='simplified_solis')


@pytest.fixture
def summer_ghi(summer_clearsky):
    """Clearsky GHI for Summer, 2020 in Albuquerque, NM."""
    return summer_clearsky['ghi']


@pytest.fixture
def summer_power_fixed(summer_clearsky, albuquerque, system_parameters):
    """Simulated power from a FIXED PVSystem in Albuquerque, NM."""
    system = pvsystem.PVSystem(**system_parameters)
    mc = modelchain.ModelChain(
        system,
        albuquerque,
        orientation_strategy='south_at_latitude_tilt'
    )
    mc.run_model(summer_clearsky)
    return mc.ac


@pytest.fixture
def summer_power_tracking(summer_clearsky, albuquerque, system_parameters):
    """Simulated power for a pvlib SingleAxisTracker PVSystem in Albuquerque"""
    system = tracking.SingleAxisTracker(**system_parameters)
    mc = modelchain.ModelChain(
        system,
        albuquerque,
        orientation_strategy='south_at_latitude_tilt'
    )
    mc.run_model(summer_clearsky)
    return mc.ac


def test_ghi_tracking_envelope_fixed(summer_ghi):
    """Clearsky GHI for has a FIXED Tracker."""
    assert system.is_tracking_envelope(
        summer_ghi,
        summer_ghi > 0,
        pd.Series(False, index=summer_ghi.index)
    ) is system.Tracker.FIXED


def test_power_tracking_envelope_fixed(summer_power_fixed):
    """Simulated system under clearsky condidtions is FIXED."""
    assert system.is_tracking_envelope(
        summer_power_fixed,
        summer_power_fixed > 0,
        pd.Series(False, index=summer_power_fixed.index)
    ) is system.Tracker.FIXED


def test_power_tracking_envelope_tracking(summer_power_tracking):
    """Simulated single axis tracker is identifified as TRACKING."""
    assert system.is_tracking_envelope(
        summer_power_tracking,
        summer_power_tracking > 0,
        pd.Series(False, index=summer_power_tracking.index)
    ) is system.Tracker.TRACKING


def test_high_clipping_unknown_tracking_envelope(summer_power_fixed):
    """If the amount of clipping is high then tracking is UNKNOWN"""
    clipping = pd.Series(False, index=summer_power_fixed.index)
    # 50% clipping
    clipping.iloc[0:len(clipping) // 2] = True
    assert system.is_tracking_envelope(
        summer_power_fixed,
        summer_power_fixed > 0,
        clipping,
        clip_max=40.0
    ) is system.Tracker.UNKNOWN


@pytest.mark.filterwarnings("ignore:invalid value encountered in",
                            "ignore:divide by zero encountered in")
def test_constant_unknown_tracking_envelope(summer_ghi):
    """A constant signal has unknown tracking."""
    constant = pd.Series(1, index=summer_ghi.index)
    assert system.is_tracking_envelope(
        constant,
        pd.Series(True, index=summer_ghi.index),
        pd.Series(False, index=summer_ghi.index),
    ) is system.Tracker.UNKNOWN


@pytest.mark.filterwarnings("ignore:invalid value encountered in",
                            "ignore:divide by zero encountered in")
def test_median_mismatch_tracking(summer_power_tracking):
    """If the median does not have the same fit as the 99.5% quantile then
    tracking is UNKNOWN."""
    power_half_tracking = summer_power_tracking.copy()
    power_half_tracking.iloc[0:100*24] = 1
    assert system.is_tracking_envelope(
        power_half_tracking,
        pd.Series(True, index=power_half_tracking.index),
        pd.Series(False, index=power_half_tracking.index),
        fit_median=False
    ) is system.Tracker.TRACKING
    assert system.is_tracking_envelope(
        power_half_tracking,
        pd.Series(True, index=power_half_tracking.index),
        pd.Series(False, index=power_half_tracking.index)
    ) is system.Tracker.UNKNOWN


@pytest.mark.filterwarnings("ignore:invalid value encountered in",
                            "ignore:divide by zero encountered in")
def test_median_mismatch_fixed(summer_power_fixed):
    """If the median does not have the same profile as the 99.5% quantile
    then tracking is UNKNOWN."""
    power_half_fixed = summer_power_fixed.copy()
    power_half_fixed.iloc[0:100*24] = 1
    assert system.is_tracking_envelope(
        power_half_fixed,
        pd.Series(True, index=power_half_fixed.index),
        pd.Series(False, index=power_half_fixed.index),
        fit_median=False
    ) is system.Tracker.FIXED
    assert system.is_tracking_envelope(
        power_half_fixed,
        pd.Series(True, index=power_half_fixed.index),
        pd.Series(False, index=power_half_fixed.index)
    ) is system.Tracker.UNKNOWN


def test_custom_tracking_envelope_thresholds(summer_power_fixed):
    """Can pass a custom set of minimal r^2 values."""
    assert system.is_tracking_envelope(
        summer_power_fixed,
        summer_power_fixed > 0,
        pd.Series(False, index=summer_power_fixed.index),
        fit_params={
            (0.5, 1.0): {'fixed': 0.9, 'tracking': 0.9, 'fixed_max': 0.9}
        }
    ) is system.Tracker.FIXED

    assert system.is_tracking_envelope(
        summer_power_fixed,
        summer_power_fixed > 0,
        pd.Series(False, index=summer_power_fixed.index),
        fit_params={
            (0.0, 1.0): {'fixed': 1.0, 'tracking': 0.8, 'fixed_max': 1.0}
        },
        fit_median=False
    ) is system.Tracker.TRACKING


@pytest.fixture(scope='module')
def albuquerque_clearsky(albuquerque):
    """One year of clearsky data in Albuquerque, NM."""
    year_hourly = pd.date_range(
        start='1/1/2020', end='1/1/2021', freq='H', tz='MST'
    )
    return albuquerque.get_clearsky(
        year_hourly,
        model='simplified_solis'
    )


def test_full_year_tracking_envelope(albuquerque_clearsky):
    """A full year of GHI should be identified as FIXED."""
    assert system.is_tracking_envelope(
        albuquerque_clearsky['ghi'],
        albuquerque_clearsky['ghi'] > 0,
        pd.Series(False, index=albuquerque_clearsky.index)
    ) is system.Tracker.FIXED


@pytest.mark.filterwarnings("ignore:invalid value encountered in",
                            "ignore:divide by zero encountered in")
def test_year_bad_winter_tracking_envelope(albuquerque_clearsky):
    """If the data is perturbed during the winter months
    is_tracking_envelope() returns Tracker.UNKNOWN."""
    winter_perturbed = albuquerque_clearsky.copy()
    winter = winter_perturbed.index.month.isin([10, 11, 12, 1, 2])
    winter_perturbed.loc[winter] = 10
    assert system.is_tracking_envelope(
        winter_perturbed['ghi'],
        albuquerque_clearsky['ghi'] > 0,
        pd.Series(False, index=albuquerque_clearsky.index)
    ) is system.Tracker.UNKNOWN