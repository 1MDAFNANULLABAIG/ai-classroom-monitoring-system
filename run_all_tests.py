"""
Automated Comprehensive Test Verification Runner
================================================
Runs all 8 verification suites covering:
1. Classroom Intelligence (36 scenarios)
2. Exam Malpractice Detection (28 scenarios)
3. Back-Facing Student Lifecycle (1 test)
4. Forensic 10s Video & Photo Evidence Validation (1 test)
5. False Positive Guardrails (10 scenarios)
6. Teacher-First State Machine (10 scenarios)
7. Advanced Attendance & Pose Profiling (25 scenarios)
8. Production E2E REST & WebSocket Integration (5 scenarios)

Total: 116 Tests.
"""

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suites = [
        loader.loadTestsFromName("test_classroom_intelligence"),
        loader.loadTestsFromName("test_exam_malpractice_monitoring"),
        loader.loadTestsFromName("test_back_facing_student"),
        loader.loadTestsFromName("verify_evidence_packages"),
        loader.loadTestsFromName("test_false_positives"),
        loader.loadTestsFromName("test_teacher_first_attendance"),
        loader.loadTestsFromName("test_advanced_attendance_monitoring"),
        loader.loadTestsFromName("test_e2e_production"),
    ]
    big_suite = unittest.TestSuite(suites)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(big_suite)

    print("\n" + "=" * 70)
    print("AI CLASSROOM SYSTEM - COMPREHENSIVE TEST REPORT")
    print("=" * 70)
    print(f"Total Tests Executed: {result.testsRun}")
    print(f"Passed:               {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures:             {len(result.failures)}")
    print(f"Errors:               {len(result.errors)}")
    print("=" * 70)

    if result.wasSuccessful():
        print(">>> ALL 116 TESTS PASSED WITH 100% SUCCESS <<<")
        sys.exit(0)
    else:
        print(">>> SOME TESTS FAILED <<<")
        sys.exit(1)
