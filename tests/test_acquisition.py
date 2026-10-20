import json
import socket
import unittest

from phoxi_yolo3d.acquisition import CaptureSchedule, SoftwareTriggerServer, TriggerBroker


class AcquisitionTests(unittest.TestCase):
    def test_manual_startup_then_no_capture_until_requested(self):
        schedule = CaptureSchedule('manual')
        self.assertEqual(schedule.take_due(0), 'startup')
        schedule.completed(1)
        self.assertIsNone(schedule.take_due(2))
        self.assertIsNone(schedule.take_due(10000))
        schedule.request_once()
        self.assertEqual(schedule.take_due(10001), 'keyboard')
        self.assertIsNone(schedule.take_due(10002))

    def test_trigger_mode_never_captures_automatically(self):
        schedule = CaptureSchedule('trigger')
        self.assertIsNone(schedule.take_due(0))
        self.assertIsNone(schedule.take_due(10000))

    def test_interval_waits_after_processing_without_catchup(self):
        schedule = CaptureSchedule('interval', 3)
        self.assertEqual(schedule.take_due(0), 'startup')
        schedule.completed(10)
        self.assertIsNone(schedule.take_due(12.9))
        self.assertEqual(schedule.take_due(13), 'interval')
        schedule.completed(30)
        self.assertIsNone(schedule.take_due(31))

    def test_pause_stops_continuous_but_allows_deliberate_step(self):
        schedule = CaptureSchedule('continuous')
        schedule.take_due(0)
        schedule.toggle_pause(1)
        self.assertIsNone(schedule.take_due(2))
        schedule.request_once()
        self.assertEqual(schedule.take_due(3), 'keyboard')
        self.assertIsNone(schedule.take_due(4))
        schedule.toggle_pause(5)
        self.assertEqual(schedule.take_due(6), 'continuous')

    def test_changing_to_manual_or_trigger_drops_automatic_schedule(self):
        schedule = CaptureSchedule('continuous')
        schedule.set_mode('manual', 0)
        self.assertIsNone(schedule.take_due(1))
        schedule.set_mode('trigger', 2)
        self.assertIsNone(schedule.take_due(3))

    def test_broker_refuses_backlog_and_cancels_pending_on_pause(self):
        broker = TriggerBroker()
        self.assertEqual(broker.submit()[1], 'not_ready')
        broker.set_enabled(True)
        first, _ = broker.submit()
        self.assertEqual(broker.submit()[1], 'busy')
        self.assertIs(broker.claim(), first)
        self.assertEqual(broker.submit()[1], 'busy')
        broker.finish(first, {'status': 'done', 'frame_id': 10})
        pending, _ = broker.submit()
        broker.set_enabled(False)
        self.assertTrue(pending.done.is_set())
        self.assertEqual(pending.result['status'], 'cancelled')
        self.assertIsNone(broker.claim())

    def test_tcp_request_gets_one_result_and_busy_is_not_queued(self):
        broker = TriggerBroker()
        broker.set_enabled(True)
        with SoftwareTriggerServer(broker, port=0) as server:
            with socket.create_connection(server.address, timeout=2) as first:
                first.sendall(b'TRIGGER\n')
                with first.makefile('r') as stream:
                    queued = json.loads(stream.readline())
                    self.assertEqual(queued['status'], 'queued')
                    request = broker.claim()
                    self.assertIsNotNone(request)
                    with socket.create_connection(server.address, timeout=2) as second:
                        second.sendall(b'TRIGGER\n')
                        with second.makefile('r') as second_stream:
                            self.assertEqual(json.loads(second_stream.readline())['status'], 'busy')
                    broker.finish(request, {'status': 'done', 'frame_id': 23})
                    done = json.loads(stream.readline())
                    self.assertEqual(done['frame_id'], 23)
                    self.assertEqual(done['request_id'], queued['request_id'])
                    self.assertIsNone(broker.claim())


if __name__ == '__main__':
    unittest.main()
