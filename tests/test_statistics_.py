# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import pytest

from main import statistics_
from testing import helpers

GET_PATH = lambda path: str((Path(__file__) / '../..' / path).resolve().absolute())
stats = statistics_.Stats()
df = pd.read_pickle(GET_PATH('testing/doc/data_ceated.pickle'))


def test_binContinuousCovariate():
    CUTOFFS = np.array([1370.0, 1950.0, 2506.0, 3160.0, 4125.0, 5028.0, 5947.4, 8408.2, 28867.8])
    FACTOR_LEVEL_STRINGS = np.array(['< 1370.0', '1370.0 - 1950.0', '1950.0 - 2506.0',
                                     '2506.0 - 3160.0', '3160.0 - 4125.0', '4125.0 - 5028.0',
                                     '5028.0 - 5947.400000000009', '5947.400000000009 - 8408.2000000',
                                     '8408.200000000012 - 28867.800000', '>= 28867.800000000017'], dtype='<U32')
    cov_vec = df['faultsDurationTime']
    bin_result = stats.binContinuousCovariate(covVec=cov_vec)
    assert isinstance(bin_result, dict)
    assert bin_result.get('cutoffs') is not None
    assert bin_result.get('factorLevelStrings') is not None
    assert len(bin_result['cutoffs']) + 1 == len(bin_result['factorLevelStrings'])
    np.testing.assert_almost_equal(CUTOFFS, bin_result['cutoffs'])
    np.testing.assert_equal(FACTOR_LEVEL_STRINGS, bin_result['factorLevelStrings'])


# @pytest.mark.skip(reason='> RUNTIME')
def test_calculateFactorLevelCut():
    dff = pd.read_csv(GET_PATH('testing/doc/calculateFactorLevelCut.csv'))
    dff.dropna(inplace=True)
    result = stats.calculateFactorLevelCut(
        cov_vector=df.faultsDurationTime,
        cutoffs=stats.binContinuousCovariate(df.faultsDurationTime)['cutoffs'], # type: ignore
        max_value=df.faultsDurationTime.max().astype(int),
        min_value=df.faultsDurationTime.min().astype(int),
    )
    assert isinstance(result, pd.Series)
    result = pd.DataFrame(result).dropna().astype(str)
    helpers.preprocess_df_split_interval(dff)
    helpers.preprocess_df_split_interval(result)
    pd.testing.assert_frame_equal(dff, result, rtol=1e-2)  # type: ignore


def test_AssignKeyColumns():
    l = pd.Index(['name', 'dateTime', 'assetId'])
    result = stats.AssignKeyColumns(
        df=pd.DataFrame({'surname': range(1, 5), 'time': range(1, 5), 'id': range(1, 5)}),
        name_col='surname', datetime_col='time', asset_id_col='id')
    assert isinstance(result, pd.DataFrame)
    assert result.columns.equals(l)

# @pytest.mark.skip(reason='> RUNTIME')


def test_computePvalueInteractionWithAssetID():
    """
    Checking that the returned element is a instance of pd.DataFrame
    """
    dff = pd.read_csv(GET_PATH('testing/doc/computePvalueInteractionWithAssetID.csv'))
    dff['faultActiveTime'] = pd.to_datetime(dff['faultActiveTime'])
    dff['createdDate'] = pd.to_datetime(dff['createdDate'])
    result = stats.computePvalueInteractionWithAssetID(data=df, faultCountsColName='numFaults', durationColName='faultsDurationTime')
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(dff, result, rtol=1e-2)  # type: ignore


# @pytest.mark.skip(reason='NEEDS FIXING')
def test_collapseFactorData():
    dff = pd.read_csv(GET_PATH('testing/doc/collapseFactorData.csv'))
    result = stats.collapseFactorData(df, assetIdColName='assetID', faultCountsColName='numFaults', covColName='faultsDurationTime')
    assert isinstance(result, pd.DataFrame)
    dff['faultsDurationTime'] -= 1  # Indexing starts from 1, atol=1 would also work.
    pd.testing.assert_frame_equal(dff, result)  # type: ignore


@pytest.mark.skip(reason='NOT IMPLEMENTED')
def test_covToFactorData():
    ...


@pytest.mark.skip(reason='NOT IMPLEMENTED')
def test_completeCollapsedData():
    ...


@pytest.mark.skip(reason='NOT IMPLEMENTED')
def test_fitRegressionModelFast():
    ...


@pytest.mark.skip(reason='NOT IMPLEMENTED')
def test_calculatePvalueAppended():
    ...


@pytest.mark.skip(reason='NOT IMPLEMENTED')
def test_significanceTestPoisModel():
    ...


@pytest.mark.skip(reason='NOT IMPLEMENTED')
def test_computePValueForVar():
    ...


@pytest.mark.skip(reason='NOT IMPLEMENTED')
def test_calculateRatioSE():
    ...


def test_robustCut():
    dff = pd.read_csv(GET_PATH('testing/doc/robustCut.csv'))
    result = stats.robustCut(vec=df.faultsDurationTime, numBins=10).astype(str)
    helpers.preprocess_df_split_interval_and_factorize(dff)
    helpers.preprocess_df_split_interval_and_factorize(result)
    pd.testing.assert_frame_equal(dff, result, atol=1)   # type: ignore


# @pytest.mark.skip(reason='> RUNTIME')
def test_removeOutliersQuantile():
    dff = pd.read_csv(GET_PATH('testing/doc/removeOutliersQuantile.csv'))['outliers'].values
    dff -= 1  # type: ignore ; Indexing in R starts from 1
    result = stats.removeOutliersQuantile(df=df, covColName='faultsDurationTime', responseColName='numFaults', cutoff=0.995)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(dff, result)


@pytest.mark.skip(reason='NOT IMPLEMENTED')
def test_removeOutliersGLMFast():
    ...


@pytest.mark.skip(reason='NOT IMPLEMENTED')
def test_compute_pValues():
    ...


@pytest.mark.skip(reason='NOT IMPLEMENTED')
def test_glm_mean_test():
    ...
