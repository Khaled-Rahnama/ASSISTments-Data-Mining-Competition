import glob
import os

import numpy as np
import pandas as pd
from pandas import Series
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
from functools import reduce
from logging import info


class Cols:
    stud_id_col = ['ITEST_id']
    label_col = ['isSTEM']
    cat_cols = ['skill', 'problemType', 'SY ASSISTments Usage']
    id_cols = ['actionId', 'problemId', 'assignmentId', 'assistmentId']
    per_action_cols = ['skill', 'problemType', 'startTime', 'endTime', 'timeTaken', 'correct', 'original', 'hint',
                       'hintCount',
                       'hintTotal', 'scaffold', 'bottomHint', 'attemptCount', 'frIsHelpRequest',
                       'frPast5HelpRequest',
                       'frPast8HelpRequest',
                       'stlHintUsed',
                       'past8BottomOut',
                       'totalFrPercentPastWrong',
                       'totalFrPastWrongCount',
                       'frPast5WrongCount',
                       'frPast8WrongCount',
                       'totalFrTimeOnSkill',
                       'timeSinceSkill',
                       'frWorkingInSchool',
                       'totalFrAttempted',
                       'totalFrSkillOpportunities',
                       'responseIsFillIn',
                       'responseIsChosen',
                       'endsWithScaffolding',
                       'endsWithAutoScaffolding',
                       'frTimeTakenOnScaffolding',
                       'frTotalSkillOpportunitiesScaffolding',
                       'totalFrSkillOpportunitiesByScaffolding',
                       'frIsHelpRequestScaffolding',
                       'timeGreater5Secprev2wrong',
                       'sumRight',
                       'helpAccessUnder2Sec',
                       'timeGreater10SecAndNextActionRight',
                       'consecutiveErrorsInRow',
                       'sumTime3SDWhen3RowRight',
                       'sumTimePerSkill',
                       'totalTimeByPercentCorrectForskill',
                       'prev5count',
                       'timeOver80',
                       'manywrong',
                       'RES_BORED',
                       'RES_CONCENTRATING',
                       'RES_CONFUSED',
                       'RES_FRUSTRATED',
                       'RES_OFFTASK',
                       'RES_GAMING'
                       ] + id_cols
    per_stud_cols = ['MCAS', 'SY ASSISTments Usage', 'NumActions', 'SchoolId', 'AveKnow',
                     'AveCarelessness'] + label_col
    conf_cols = ['confidence(BORED)', 'confidence(CONCENTRATING)', 'confidence(CONFUSED)', 'confidence(FRUSTRATED)',
                 'confidence(OFF TASK)', 'confidence(GAMING)']
    ave_res_cols = ['AveCorrect', 'AveResBored', 'AveResEngcon', 'AveResConf', 'AveResFrust', 'AveResOfftask',
                    'AveResGaming']
    excluded_cols = ['prev5count', 'Prev5count', 'timeSinceSkill', 'sumTime3SDWhen3RowRight', 'Ln-1',
                     'Ln'] + ave_res_cols + conf_cols

    per_stud_cols_cat = list(set(per_stud_cols).intersection(set(cat_cols)))
    per_action_cols_cat = list(set(per_action_cols).intersection(set(cat_cols)))

    paper_suggested_cols = ["attemptCount", "bottomHint", "consecutiveErrorsInRow", "correct",
                            "endsWithAutoScaffolding",
                            "endsWithScaffolding", "frIsHelpRequest", "frIsHelpRequestScaffolding",
                            "frPast5HelpRequest",
                            "frPast5WrongCount", "frPast8HelpRequest", "frPast8WrongCount", "frTimeTakenOnScaffolding",
                            "frTotalSkillOpportunitiesScaffolding", "frWorkingInSchool", "helpAccessUnder10Sec",
                            "helpAccessUnder1Sec", "helpAccessUnder2Sec", "helpAccessUnder5Sec",
                            "helpOnFirstAttemptAndTimeLess2Sec", "hint", "hintCount", "hintTotal", "original",
                            "past8BottomOut",
                            "percentCorrectPerSkill", "prevHintThisWrong", "responseIsChosen", "responseIsFillIn",
                            "scaffold",
                            "stlHintUsed", "sumHelp", "sumofRightPerSkill", "sumRight",
                            "sumSameSkillWrongonFirstAttempt",
                            "sumTime3SDWhen3RowRight", "sumTime5SDWhen5RowRight", "sumTimePerSkill",
                            "timeGreater10AndPrevActionWrong", "timeGreater10SecAndNextActionRight",
                            "timeGreater10SecPrevActionHelpOrBug", "timeGreater5Secprev2wrong", "timeSinceSkill",
                            "timeTaken",
                            "totalFrAttempted", "totalFrPastWrongCount", "totalFrPercentPastWrong",
                            "totalFrSkillOpportunities",
                            "totalFrSkillOpportunitiesByScaffolding", "totalFrTimeOnSkill",
                            "totalTimeByPercentCorrectForskill"]


class Preprocessing:
    data_path = r"Dataset"

    data_file_name = "student_log_*.csv"
    label_file_name = r'training_label.csv'
    test_file_name = r'validation_test_label.csv'

    col_dtype = {'totalFrSkillOpportunitiesByScaffolding': np.float32, 'totalFrTimeOnSkill': np.float32}

    raw_dataset = None

    per_stud_dataset = None
    per_action_dataset = None
    per_stud_dataset_cat = None
    per_action_dataset_cat = None
    per_stud_dataset_enc = None
    per_action_dataset_enc = None
    per_action_dataset_cat_summ = None
    per_action_dataset_summ = None

    label_dateset = None
    test_dataset = None

    def load_data(self, time_gap=None,
                  encode=False,
                  include_cat_data=False,
                  return_val_tst_set=False):

        data_files = glob.glob(os.path.join(self.data_path, self.data_file_name))
        label_file = glob.glob(os.path.join(self.data_path, self.label_file_name))[0]
        test_file = glob.glob(os.path.join(self.data_path, self.test_file_name))[0]

        self.raw_dataset = pd.concat((pd.read_csv(f, dtype=self.col_dtype) for f in data_files))

        # 'AveCorrect' column is already in dataset file and it can be dropped from label/test file
        self.label_dataset = pd.read_csv(label_file).drop_duplicates()
        if any(["AveCorrect" in col for col in self.label_dataset.columns]):
            self.label_dataset = self.label_dataset.drop("AveCorrect", axis=1)

        self.test_dataset = pd.read_csv(test_file).drop_duplicates()
        if any(["AveCorrect" in col for col in self.test_dataset.columns]):
            self.test_dataset = self.test_dataset.drop("AveCorrect", axis=1)

        if return_val_tst_set:
            self.test_dataset['isSTEM'] = -1
            x = pd.merge(self.raw_dataset, self.test_dataset, on=Cols.stud_id_col).drop(Cols.excluded_cols, axis=1)
        else:
            x = pd.merge(self.raw_dataset, self.label_dataset, on=Cols.stud_id_col).drop(Cols.excluded_cols, axis=1)
        per_stud_dataset = x[
            list(set(Cols.per_stud_cols).difference(set(Cols.excluded_cols))) + Cols.stud_id_col].drop_duplicates()
        per_action_dataset = x[list(set(Cols.per_action_cols).difference(set(Cols.excluded_cols))) + Cols.stud_id_col]

        self.per_stud_dataset_cat = per_stud_dataset[Cols.per_stud_cols_cat + Cols.stud_id_col]
        self.per_action_dataset_cat = per_action_dataset[Cols.per_action_cols_cat + Cols.stud_id_col + ['actionId']]

        self.per_stud_dataset = per_stud_dataset.drop(Cols.per_stud_cols_cat, axis=1)
        self.per_action_dataset = per_action_dataset.drop(Cols.per_action_cols_cat, axis=1)

        self.per_stud_dataset_cat.name = 'per_stud_dataset_cat'
        self.per_action_dataset_cat.name = 'per_action_dataset_cat'
        self.per_stud_dataset.name = 'per_stud_dataset'
        self.per_action_dataset.name = 'per_action_dataset'

        if encode:
            self.per_action_dataset_enc = self.__encode(self.per_action_dataset_cat, Cols.per_action_cols_cat)
            self.per_stud_dataset_enc = self.__encode(self.per_stud_dataset_cat, Cols.per_stud_cols_cat)
            self.per_action_dataset_enc.name = 'per_action_dataset_enc'
            self.per_stud_dataset_enc.name = 'per_stud_dataset_enc'

        # dropping per_student columns if there are any of those in current columns of dataset
        # and summarizing rest of the columns follow joining the per_student columns
        if time_gap:

            if self.per_action_dataset_enc is not None:
                self.per_action_dataset_cat_summ = self.__load_summarized(self.per_action_dataset_enc, time_gap)
                self.per_action_dataset_cat_summ.name = 'per_action_dataset_cat_summ'

            # removing id cols from dataset
            per_action_dataset_noidcols = self.per_action_dataset.drop(Cols.id_cols, axis=1)
            self.per_action_dataset_summ = self.__load_summarized(per_action_dataset_noidcols, time_gap)
            self.per_action_dataset_summ.name = 'per_action_dataset_summ'

        prepared_dataset = None
        merged_candidate = None

        # building all combination of parameter flags and merging required data frames
        if include_cat_data:
            if encode:
                if time_gap:
                    merged_candidate = [self.per_action_dataset_cat_summ, self.per_action_dataset_summ,
                                        self.per_stud_dataset_enc, self.per_stud_dataset]
                    print('prepared dataset contains: %s + %s + %s + %s' % (
                        'per_action_dataset_cat_summ', 'per_action_dataset_summ', 'per_stud_dataset_enc',
                        'per_stud_dataset'))
                else:
                    merged_candidate = [self.per_action_dataset, self.per_action_dataset_enc]
                    print('prepared dataset contains: %s + %s' % ('per_action_dataset', 'per_action_dataset_enc'))
            else:
                if time_gap:
                    merged_candidate = [self.per_action_dataset_summ, self.per_stud_dataset_cat,
                                        self.per_stud_dataset]
                    print('prepared dataset contains: %s + %s + %s' % (
                        'per_action_dataset_summ', 'per_stud_dataset_cat', 'per_stud_dataset'))
                else:
                    raise NotImplementedError
        else:
            if encode:
                raise AssertionError
            else:
                if time_gap:
                    merged_candidate = [self.per_stud_dataset, self.per_action_dataset_summ]
                    print("prepared dataset contains: %s + %s" % ('per_stud_dataset', 'per_action_dataset_summ'))
                else:
                    prepared_dataset = self.per_action_dataset
                    print('prepared dataset contains: %s' % 'per_action_dataset')
                    # TODO write with dataframe.name
        if merged_candidate:
            prepared_dataset = reduce(lambda left, right: pd.merge(left, right, on=Cols.stud_id_col), merged_candidate)

        # TODO do clean up messy data...
        # x[x < 0] = np.nan
        # x.fillna(0, inplace=True)

        try:
            index = [prepared_dataset.ITEST_id, prepared_dataset.seq_ix]
            prepared_dataset = prepared_dataset.drop(["ITEST_id", "seq_ix"], axis=1)
        except KeyError:
            print("Dataset contains only one index (ITEST_id)!")
            index = prepared_dataset.ITEST_id
            prepared_dataset = prepared_dataset.drop("ITEST_id", axis=1)

        prepared_dataset.index = index

        try:
            y = prepared_dataset[Cols.label_col]
            y.index = prepared_dataset.index.get_level_values("ITEST_id")
            prepared_dataset = prepared_dataset.drop(Cols.label_col, axis=1)
        except KeyError:
            y = None
            print("Warning! There is no label column in this setting!")

        return prepared_dataset, y


    def __encode(self, x, cat_cols):
        enc_x = pd.get_dummies(x, columns=cat_cols)
        return enc_x

    def __summarize_seq(self, stud_seq, time_gap):
        """
        Summarize or builds sessions on student actions by grouping contagious actions with time distance less than time_gap
        :param stud_seq: a data-frame containing a series of actions for a student sorted by the 'startTime' ascending
        :param time_gap: a time distant threshold in seconds between each session
        :return: a summarized data-frame with rows as sessions where actions of session have been summarized by row Max, Min, Mean, Sum functions
        """

        current_action_endtime = Series(stud_seq['endTime'])
        next_action_starttime = Series(stud_seq['startTime']).shift(-1)

        diff = (next_action_starttime - current_action_endtime).dropna()

        session_end_ix = np.where(diff > time_gap)[0]
        session_start_end_ix = np.insert(session_end_ix, 0, 0)
        session_start_end_ix = np.insert(session_start_end_ix, len(session_start_end_ix), len(stud_seq) - 1)
        session_lengths = np.diff(session_start_end_ix)
        session_lengths[0] = session_lengths[
                                 0] + 1  # correct the length of first session by adding 1 because of zero index
        session_index = np.repeat(np.array(range(len(session_start_end_ix) - 1)), session_lengths)

        sorted_seq_with_sess_ix = stud_seq.assign(seq_ix=session_index)
        min_seq = sorted_seq_with_sess_ix.groupby('seq_ix').min().add_prefix('min_')
        max_seq = sorted_seq_with_sess_ix.groupby('seq_ix').max().add_prefix('max_')
        mean_seq = sorted_seq_with_sess_ix.groupby('seq_ix').mean().add_prefix('mean_')
        sum_seq = sorted_seq_with_sess_ix.groupby('seq_ix').sum().add_prefix('sum_')

        summarized_features = pd.concat([min_seq, max_seq, mean_seq, sum_seq], axis=1)

        return summarized_features

    def __load_summarized(self, x, time_gap):
        """
        Divide the student actions into sub-sequences (sessions) and summarize them using Min, Max, Mean, Sum
        :param x: a data-frame consisting of sequence of all actions for all students
        :param time_gap: the time gap threshold  in seconds for separating consecutive actions of the student
        :return: # a summarized dataframe with a new set of summarized features including a seq_ix column as the session id
        """
        # assumes that input contains "startTime" and "ITEST_id" columns
        assert (all([col in [col for col in x.columns] for col in ["startTime", "ITEST_id"]]))

        temp_x = []

        for stud_id, stud_seq in x.groupby('ITEST_id'):
            stud_seq = stud_seq.drop(['ITEST_id'], axis=1)
            sorted_stud_seq = stud_seq.sort_values('startTime')
            summarized_stud_seq = self.__summarize_seq(sorted_stud_seq, time_gap)
            summarized_stud_seq['seq_ix'] = summarized_stud_seq.index
            summarized_stud_seq['ITEST_id'] = stud_id
            temp_x.append(summarized_stud_seq)

        temp_x = pd.concat(temp_x)

        return temp_x

    def save_datasets(self, datasets):
        for dataset in datasets:
            if dataset is not None:
                dataset.to_csv("Preprocessed Dataset\%s.csv" % dataset.name, index=False)
