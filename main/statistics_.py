# -*- coding: utf-8 -*-
from statistics import mean

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy.contrasts import Sum, Treatment
from scipy.stats import bootstrap, norm, poisson


class Stats():
    @staticmethod
    def ratioFunc(df, i, *args, **kwargs) -> float:
        '''statistic for scipy.stats.bootstrap'''
        return df[1, i].sum() / df[0, i].sum()

    @staticmethod
    def binContinuousCovariate(covVec: pd.Series, numBinsByUser: int = 10) -> dict[str, np.ndarray]:
        covVec = covVec[~np.isnan(covVec)]
        q = np.linspace(0, 1, numBinsByUser, endpoint=False)[1:]
        cutoffs = np.quantile(a=covVec, q=q)
        factorLevelStrings = np.repeat(np.nan, cutoffs.size + 1).astype('str')
        for i in range(factorLevelStrings.size):
            if i == 0:
                factorLevelStrings[i] = f'< {cutoffs[i]}'
            elif i == factorLevelStrings.size - 1:
                factorLevelStrings[i] = f'>= {cutoffs[i - 1]}'
            else:
                factorLevelStrings[i] = f'{cutoffs[i - 1]} - {cutoffs[i]}'
        return dict(cutoffs=cutoffs, factorLevelStrings=factorLevelStrings)

    @staticmethod
    def calculateFactorLevelCut(cov_vector: pd.Series, cutoffs: pd.Series, min_value: int, max_value: int) -> pd.Series:
        bins = np.sort(np.append(cutoffs, np.array([min_value, max_value])))
        cov_factor = pd.cut(cov_vector, bins=bins)
        return cov_factor

    @staticmethod
    def AssignKeyColumns(df: pd.DataFrame, name_col: str = 'name',
                         datetime_col: str = 'dateTime', asset_id_col: str = 'assetId') -> pd.DataFrame:
        return df.rename(columns={name_col: 'name', datetime_col: 'dateTime', asset_id_col: 'assetId'})

    @staticmethod
    def computePvalueInteractionWithAssetID(data: pd.DataFrame, faultCountsColName: str, durationColName: str) -> pd.DataFrame:
        data_save_col_names = data.columns
        data_save = data.copy()
        if not durationColName:
            faultCounts = data[faultCountsColName]
            meanFaultCounts = faultCounts.mean()
            data_save['coeffvalue'] = faultCounts / meanFaultCounts
            data_save['SEcoeff'] = 0
            if data_save.shape[0] > 20:
                data_save['pvalue'] = (1 - data_save['coeffvalue'].rank() / data_save.shape[0])
            else:
                data_save['pvalue'] = (1 - poisson.cdf(k=faultCounts, mu=meanFaultCounts))
        else:
            faultCounts = data[faultCountsColName]
            faultDuration = data[durationColName]
            mean_occurence_rate = faultCounts.sum() / faultDuration.sum()
            data_save['occurenceRate'] = faultCounts / faultDuration
            data_save['coeffvalue'] = (faultCounts / faultDuration / mean_occurence_rate)
            data_save['SEcoeff'] = 0
            if data_save.shape[0] > 20:
                data_save['pvalue'] = (1 - data_save['coeffvalue'].rank() / data_save.shape[0])
            else:
                data_save['poissonMean'] = mean_occurence_rate * faultDuration
                data_save['pvalue'] = (1 - poisson.cdf(k=faultCounts, mu=data_save['poissonMean']))
        data_save_col_names = np.append(data_save_col_names, np.array(['coeffvalue', 'SEcoeff', 'pvalue']))
        df = data_save[data_save_col_names]
        return df

    @staticmethod
    def collapseFactorData(data: pd.DataFrame, assetIdColName: str, faultCountsColName: str, covColName: str) -> pd.DataFrame:
        cov_factor, _ = pd.factorize(data[covColName], sort=True)
        asset_id = data[assetIdColName]
        num_faults = data[faultCountsColName]
        uncollapsedData = pd.DataFrame(dict(covFactor=cov_factor, assetID=asset_id, numFaults=num_faults))
        uncollapsedData = uncollapsedData.sort_values(by=['covFactor', 'assetID'], ascending=(True, True))
        collapsedData = uncollapsedData.groupby(['covFactor', assetIdColName]).apply(lambda x: x['numFaults'].sum())
        collapsedData = collapsedData.to_frame()
        collapsedData.columns = ['numFaults']
        collapsedData = collapsedData.reset_index()
        collapsedData = collapsedData[['covFactor', assetIdColName, faultCountsColName]]
        collapsedData.columns = [covColName, assetIdColName, faultCountsColName]
        return collapsedData

    def covToFactorData(self, data: pd.DataFrame, assetIdColName: str, faultCountsColName: str,
                        covColName: str, cutoffs: np.ndarray, factorLevels: np.ndarray) -> pd.DataFrame:
        raise NotImplementedError
        cutoffs = np.sort(cutoffs, axis=None)
        covVec = data[covColName]
        covFactor = np.apply_along_axis(lambda x: np.sum(x >= cutoffs) + 1, axis=0, arr=covVec)
        uncollapsedData = pd.concat([pd.DataFrame({covColName: factorLevels[covFactor]}), data[assetIdColName], data[faultCountsColName]])
        collapsedData = self.collapseFactorData(uncollapsedData, assetIdColName, faultCountsColName, covColName)
        return collapsedData

    def completeCollapsedData(self, collapsedData: pd.DataFrame, factorOfInterestName: str,
                              factorLevels: np.ndarray, faultCountsColName: str, assetIdColName: str):
        addedRows = None
        levelForthisFault = collapsedData[factorOfInterestName].astype(int).unique()
        factorLevels = factorLevels.astype(int)
        missingLevels = factorLevels[~(np.isin(factorLevels, levelForthisFault))]
        numpTmp = missingLevels.size
        if numpTmp:
            addedRows = pd.DataFrame()
            addedRows['factor'] = missingLevels
            addedRows['assetId'] = 'LOCOMOTIVE_0000'
            addedRows['num_faults'] = numpTmp
            addedRows.columns = [factorOfInterestName, assetIdColName, faultCountsColName]
        returnedData = pd.concat((collapsedData, addedRows)) # type: ignore
        returnedData = returnedData.sort_values(by=[factorOfInterestName, assetIdColName], ascending=(True, True))
        return returnedData

    @staticmethod
    def fitRegressionModelFast(faultCounts: None, factorOfInterest, faultDuration, numLevels: int, factorLevels, family: str):
        raise NotImplementedError
        if not faultDuration:
            dataSpeedGLM = pd.DataFrame({'faultCounts': faultCounts, 'factorOfInterest': factorOfInterest, 'faultDuration': faultDuration})
        else:
            dataSpeedGLM = pd.DataFrame({'faultCounts': faultCounts, 'factorOfInterest': factorOfInterest})
        if family == 'poisson':
            if not faultDuration:
                reg = smf.glm('faultCounts~factorOfInterest', data=dataSpeedGLM, family=sm.families.Poisson(link='log')).fit()
            else:
                # this should be checked with a real example
                # ref: https://stackoverflow.com/questions/42194356/offset-argument-in-statsmodels-api-glm
                reg = smf.glm('faultCounts~1+factorOfInterest', data=dataSpeedGLM,
                              offset=np.log(dataSpeedGLM['faultDuration']), family=sm.families.Poisson(link='log')).fit()
        elif family == 'gaussian':
            reg = smf.glm('faultCounts~factorOfInterest', data=dataSpeedGLM, family=sm.families.Gaussian()).fit()
        else:
            raise ValueError(f'Currently, the function does not supper tihs familiy: {family}')
        # ref: https://stackoverflow.com/questions/51734180/converting-statsmodels-summary-object-to-pandas-dataframe
        regSummary = pd.read_html(reg.summary().tables[1].as_html(), header=0, index_col=0)[0]
        estimateLastLevel = np.sum(0, np.repeat(-1, numLevels - 1)) * np.asarray(regSummary['coeff'])
        # ref: https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.html#statsmodels.genmod.generalized_linear_model.GLM
        SELastLevel = np.sqrt(np.r_[0, np.repeat(-1, numLevels - 1)]) @ reg.normalized_cov_params @ np.r_[0, np.repeat(-1, numLevels - 1)]
        zvalueLastLevel = estimateLastLevel / SELastLevel
        # ref: https://www.statsmodels.org/devel/glm.html
        zvalue = np.r_[regSummary[2:numLevels, regSummary['z']], zvalueLastLevel]
        coeffValue = None
        if family == 'poisson':
            coeffValue = np.r_[np.exp(regSummary['coeff'].loc[2:numLevels], np.exp(estimateLastLevel))]
            # variance covariance matrix using pandas
            estVariance = np.r_[np.diag(reg.normalized_cov_params)[2:numLevels], SELastLevel ** 2]
            SEcoeff = np.sqrt(coeffValue * estVariance * coeffValue)
        elif family == 'gaussian':
            coeffValue = np.r_[regSummary['coeff'].loc[2:numLevels], estimateLastLevel]
            SEcoeff = np.r_[regSummary['coeff'].loc[2:numLevels], SELastLevel]
        else:
            raise ValueError(f'Currently, the function does not support this family: {family}')
        pvalueMatrix = pd.DataFrame({
            'factorOfInterest': factorLevels,
            f'zvalue{family.upper()}': zvalue,
            f'coeffvalue{family.upper()}': coeffValue,
            f'SEcoeff{family.upper()}': SEcoeff
        })
        return pvalueMatrix

    @staticmethod
    def calculatePvalueAppended(pvalueMatrix: pd.DataFrame, zvalue, testingSide: str):
        # ref: https://stackoverflow.com/questions/24695174/python-equivalent-of-qnorm-qf-and-qchi2-of-r
        if testingSide == 'right':
            pvalueMatrix['pvaluePoissonRightSided'] = 1 - norm.cdf(zvalue)
            return pvalueMatrix
        if testingSide == 'left':
            pvalueMatrix['pvaluePoissonLeftSided'] = norm.cdf(zvalue)
            return pvalueMatrix
        pvalueMatrix['pvaluePoissonTwoSided'] = np.where(zvalue > 0, (1 - norm.cdf(zvalue)) * 2, norm.cdf(zvalue) * 2)
        return pvalueMatrix

    def significanceTestPoisModel(self, dataFinal: pd.DataFrame, testFactorColName: str, durationColName: str,
                                  method: str = 'poisson', testingSide='right', countColName='num_faults'):
        factorOfInterest = pd.Categorical(dataFinal[testFactorColName])  # type: ignore
        faultCounts = dataFinal[countColName]
        faultDuration = None
        if not durationColName:
            faultDuration = dataFinal[durationColName]
        factorLevels = factorOfInterest.categories.tolist()
        numLevels = len(factorLevels)
        if numLevels > 10000:
            raise ValueError(f'The number of levels of the factor of interest is larger than 10000!\nNo hypothesis tests are performed')
        if numLevels >= 2 and dataFinal.shape[0] > numLevels:
            if numLevels > 2:
                contrast = Sum().code_without_intercept(factorOfInterest)  # contr.sum
            else:
                contrast = Treatment().code_without_intercept(
                    factorOfInterest)  # contr.treatment
                testingSide = 'two-sided'
            pvalueMatrix = pd.DataFrame(self.fitRegressionModelFast(faultCounts, factorOfInterest, faultDuration, numLevels, factorLevels, family=method))
            pvalueMatrix = pd.DataFrame(self.calculatePvalueAppended(pvalueMatrix, pvalueMatrix[zvalue], testingSide))
            pvalueMatrix.set_axis(pd.Series(np.linspace(1, pvalueMatrix.shape[0])))
        else:
            raise ValueError(f'Cannot run statistical test if the number of observations is 1 for each group')
        return pvalueMatrix

    def computePvalueForVar(self, data: pd.DataFrame, covColName: str, durationColName: str, isContinuous: bool = False,
                            assetIdColName: str = 'assetId', faultCountsColNAme: str = 'num_faults', isCompleteData: bool = False,
                            testingSide: str = 'right', modelType: str = 'poisson', removeOutliersOption: int = 0,
                            removeOutliersThreshold: float = 0.995):

        if modelType != 'poisson':
            raise ValueError(f'Currently, the function does not support this family: {modelType}')

        interactionWithAssetId = data.shape[0] == len(data[covColName].unique())  # retaining unique values
        if interactionWithAssetId:
            data_save = self.computePvalueInteractionWithAssetID(data, faultCountsColNAme, durationColName)
            data_save['p_value'] = data_save['p_value'].replace(np.nan, 1)
            try:
                data_save['coeffvalue'] = data_save['coeffvalue'].replace(np.nan, 1)
                return data_save
            except Exception:
                pass
        else:
            data_save = data.copy()
            data = data[[covColName, assetIdColName, faultCountsColNAme, durationColName]]
            if not isContinuous:
                data[covColName] = data[covColName].astype('category')
                if data.shape[0] > data[assetIdColName].unique().shape[0]:
                    raise ValueError(f'Duplicated rows for factor level and assetId combination')
                else:
                    collapseData = data.copy()
            else:
                raise ValueError(f'Currently the function does not support the option: isContinuous=True')
            if isCompleteData == True:
                raise ValueError(f'Currently the function does not support the option: isContinuous=True')
            outliersIndices = np.nan
            if removeOutliersOption == 1:
                if np.isnan(durationColName):
                    tempData = pd.DataFrame({'v1': collapseData[covColName], 'v2': [collapseData[faultCountsColNAme]]})
                else:
                    tempData = pd.DataFrame({
                        'v1': collapseData[covColName],
                        'v2': collapseData[faultCountsColNAme] / collapseData[durationColName]
                    })
                tempData = tempData.rename(columns={'v1': covColName, 'v2': 'faultOccurrence'})
                outliersIndices = self.removeOutliersQuantile(
                    df=tempData, covColName=covColName, responseColName='faultOccurrence', cutoff=removeOutliersThreshold
                )
            elif removeOutliersOption == 2:
                if np.isnan(durationColName):
                    faultDurationVal = np.nan
                else:
                    faultDurationVal = collapseData[durationColName].values
                print(f'faultDurationVal be: {faultDurationVal}')
                outliersIndices = np.asarray(self.removeOutliersGLMfast(
                    faultCounts=collapsedData[faultCountsColName].values, factorOfInterest=collapsedData[covColName].values,
                    faultDuration=faultDurationVal, family=modelType, cutoff=removeOutliersThreshold))
                if np.isnan(outliersIndices) == False:
                    # in both cases performing discriminative filtering based on the outliersIndices array
                    collapseData = collapseData.iloc[collapseData.index.astype('object').isin(outliersIndices.astype('object')), :]
                    data_save = data_save.iloc[data_save.index.astype('object').isin(outliersIndices.astype('object')), :]
                tmpRes = self.significanceTestPoisModel(
                    collapsedData, countColName=faultCountsColName, testFactorColName=covColName,
                    urationColName=durationColName, method=modelType, testingSide=testingSide
                )
                # line 122 of R file
                # this was the intuition since `:=` can have a few usage and there's no documentation explaining what it does in the code
                data_save['coeffvalue'] = tmpRes[data_save[covColName].isin(tmpRes[factorOfInterest]), 3]
                data_save['SEcoeff'] = tmpRes[data_save[covColName].isin(tmpRes[factorOfInterest]), 4]
                data_save['p_value'] = tmpRes[data_save[covColName].isin(tmpRes[factorOfInterest]), 5]
                data_save['pvalue'] = data_save['pvalue'].replace(np.nan, 1)
                if modelType == 'poisson':
                    data_save['coeffvalue'] = data_save['coeffvalue'].replace(np.nan, 1)
                elif modelType == 'gaussian':
                    data_save['coeffvalue'].replace(np.nan, 0)
                else:
                    raise ValueError(f'Currently, the funciton does not support this family: {modelType}')
                return pd.DataFrame(dict(testingResults=tmpRes, data=data_save))

    def calculateRatioSE(self, x: np.ndarray, y: np.ndarray, method: str = 'both'):
        if x.size != y.size:
            raise ValueError(f'The lengths of two pass-in vector differ')
        if np.sum(x == 0) > 0:
            raise ValueError(f'The denominator has a zero element')

        indNA = np.r_[[i for i, v in enumerate(x) if np.isnan(v)], [i for i, v in enumerate(y) if np.isnan(v)]]
        if indNA.size:
            x = np.delete(x, indNA)
            y = np.delete(y, indNA)
        if len(np.unique(x)) == 1 & len(np.unique(y)) == 1:
            return dict(SEBoot=0, SEDelta=0)

        SEDelta = np.nan
        if method in ('both', 'delta'):
            n = x.size
            mx = x.mean()
            my = y.mean()
            vxbar = np.var(x) / n
            vybar = np.var(y) / n
            vrat = (my**2 / mx**4) * vxbar + (1 / mx**2) * vybar - (2 * (my / mx**3) * np.cov(x, y)[1, 0] / n)
            SEDelta = np.sqrt(vrat)

        SEBoot = np.nan
        if method in ('both', 'bootstrap') or not np.isfinite(SEDelta):
            data = pd.DataFrame({'x': x, 'y': y})
            # line 356 bootstrap sampling in python, ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
            BootObj = bootstrap((data,), self.ratioFunc, n_resamples=500)
            SEBoot = BootObj.standard_error
        if not np.isfinite(SEDelta):
            SEDelta = SEBoot
        return dict(SEBoot=SEBoot, SEDelta=SEDelta)

    @staticmethod
    def robustCut(vec, numBins=10):
        vec = vec.astype(int)
        indNoNA = vec.index[~np.isnan(vec)]
        vec = vec.loc[~np.isnan(vec)]
        length = vec.size

        meanRobust = np.median(vec)
        sdRobust = np.median(abs(vec - np.median(vec))) / 0.675
        upperCutoff = meanRobust + sdRobust * 3
        lowerCutoff = meanRobust - sdRobust * 3
        brks = None
        if np.sum(vec > upperCutoff) > 0 and np.sum(vec < lowerCutoff) > 0:
            lowerCutoff = vec.loc[vec >= lowerCutoff].min()
            upperCutoff = vec.loc[vec <= upperCutoff].max()
            brks = np.r_[vec.min(), np.linspace(lowerCutoff, upperCutoff, numBins - 1), vec.max()]
        if np.sum(vec > upperCutoff) > 0 and np.sum(vec < lowerCutoff) == 0:
            lowerCutoff = vec.min()
            upperCutoff = vec.loc[vec <= upperCutoff].max()
            brks = np.r_[vec.min(), np.linspace(lowerCutoff, upperCutoff, numBins), vec.max()]
        if np.sum(vec > upperCutoff) == 0 and np.sum(vec < lowerCutoff) > 0:
            lowerCutoff = vec.loc[vec >= lowerCutoff].min()
            upperCutoff = vec.max()
            brks = np.r_[vec.min(), np.linspace(lowerCutoff, upperCutoff, numBins)]
        if np.sum(vec > upperCutoff) == 0 and np.sum(vec < lowerCutoff) == 0:
            lowerCutoff = vec.min()
            upperCutoff = vec.max()
            brks = np.linspace(lowerCutoff, upperCutoff, numBins + 1)
        result = pd.cut(vec, bins=brks, include_lowest=True, precision=3)
        return result.to_frame()

    @staticmethod
    def removeOutliersQuantile(df: pd.DataFrame, covColName: str, responseColName: str, cutoff: float = 0.995) -> np.ndarray:
        temp_data = pd.DataFrame()
        temp_data['rowID'] = df.index
        temp_data['factorOfInterest'] = pd.Categorical(df[covColName])  # type: ignore
        temp_data['faultsOcurrence'] = df[responseColName]
        factor_levels = temp_data['factorOfInterest'].cat.categories  # type: ignore
        outlierIndicesQuantile = []
        for i in range(factor_levels.size):
            temp = temp_data.loc[temp_data['factorOfInterest'] == factor_levels[i], :]
            threshold = np.quantile(a=temp['faultsOcurrence'], q=cutoff)
            outliers = temp[temp['faultsOcurrence'] > threshold].dropna()['rowID'].values
            if outliers.size:  # type: ignore
                outlierIndicesQuantile.append(outliers[0])
        outlierIndicesQuantile = np.array(outlierIndicesQuantile)
        return outlierIndicesQuantile

    @staticmethod
    def removeOutliersGLMfast(faultCounts, factorOfInterest, faultDuration: None, family='poisson', cutoff=0.995):
        print(f'passed faultDuration be: {faultDuration}')
        if not faultDuration:
            df = pd.DataFrame({'faultCounts': faultCounts, 'factorOfInterest': factorOfInterest})
        else:
            df = pd.DataFrame({'faultCounts': faultCounts, 'factorOfInterest': factorOfInterest, 'faultDurations': faultDuration})
        contrast_sum = Sum().code_without_intercept(factorOfInterest)
        if family == 'poisson':
            if faultDuration is None:
                reg = smf.glm('faultCounts ~ factorOfInterest', data=df, family=sm.families.Poisson(link='log')).fit()
            else:
                print(f'got to reg, and faultDuration is: {df.columns.tolist()}\n{df.head()}\nSummary\n{df.describe()}')
            reg = smf.glm('faultCounts~1+factorOfInterest', data=df, offset=np.log(df['faultDuration']),
                          family=sm.families.Poisson(link='log')).fit()
            print(f'finished reg {reg.summary()}')
            regSummary = pd.read_html(reg.summary().tables[1].as_html(), header=0, index_col=0)[0]
            fittedVals = np.r_[regSummary['coef']]
            pear_res = (faultCounts - fittedVals) / np.sqrt(fittedVals)

        elif family == 'gaussian':
            reg = smf.glm('faultCounts ~ factorOfInterest', data=df, family=sm.families.Gaussian()).fit()
            regSummary = pd.read_html(reg.summary().tables[1].as_html(), header=0, index_col=0)[0]
            fittedVals = np.r_[regSummary['coef']]
            pear_res = faultCounts - fittedVals
        else:
            raise ValueError(f'Currently, the function does not support this family : {family}')
        outlierIndicesPearsonResid = np.where(pear_res > np.quantile(pear_res, q=cutoff))
        return outlierIndicesPearsonResid

    def compute_pValues_af(*args, **kwargs):
        ...

    def glm_mean_test(*args, **kwargs):
        ...
