import unittest
from tools.association_validation import evaluate,assignments


class ValidationMetricsTests(unittest.TestCase):
    def test_unreviewed_rows_are_not_truth(self):
        self.assertEqual(evaluate([dict(review_status='UNREVIEWED')],{})['status'],'AWAITING_INDEPENDENT_REVIEW')

    def test_switch_and_reconnect_metrics_are_not_confidence_scores(self):
        truth=[dict(frame=f,detection_id='0',review_status='VERIFIED',kind='target',object_id='A') for f in (1,2,5,6)]
        result=evaluate(truth,{(1,'0'):'one',(2,'0'):'one',(5,'0'):'one',(6,'0'):'two'})
        self.assertEqual(result['identity_switches'],1)
        self.assertEqual(result['same_label_reconnects'],1)
        self.assertEqual(result['extra_labels_per_object_sum'],1)

    def test_uncertain_identity_stays_unassigned(self):
        group=[dict(weight=.6,targets=[('T:a',(1,'0'))]),dict(weight=.4,targets=[('T:b',(1,'0'))])]
        self.assertEqual(assignments([group]),{})

    def test_duplicate_truth_cannot_inflate_metrics(self):
        row=dict(frame=1,detection_id='0',review_status='VERIFIED',kind='target',object_id='A')
        with self.assertRaises(ValueError):evaluate([row,row],{})

    def test_verified_non_target_association_is_counted(self):
        rows=[dict(frame=1,detection_id='0',review_status='VERIFIED',kind='target',object_id='A'),
              dict(frame=1,detection_id='1',review_status='VERIFIED',kind='star',object_id='')]
        result=evaluate(rows,{(1,'0'):'a',(1,'1'):'b'})
        self.assertEqual(result['target_assignments_on_verified_non_targets'],1)

    def test_non_target_only_review_is_still_valid_evidence(self):
        rows=[dict(frame=1,detection_id='0',review_status='VERIFIED',kind='clutter',object_id='')]
        result=evaluate(rows,{(1,'0'):'a'})
        self.assertEqual(result['status'],'EVALUATED_REVIEWED_SUBSET')
        self.assertEqual(result['target_assignments_on_verified_non_targets'],1)
