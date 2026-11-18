from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import threading
import time
import uuid

from ..storage import _json_safe, utc_stem
from ..config import project_path
from .contract import STEPS, check_files, digest, make_job, timestamp_age


class StopAfterCycle(Exception):
    pass


class Lines:
    def __init__(self, conn):
        self.conn, self.buffer = conn, bytearray()

    def send(self, text):
        if '\n' in text or '\r' in text or not text.isascii():
            raise ValueError('INVALID_OUTGOING_LINE')
        self.conn.settimeout(5)
        self.conn.sendall((text+'\n').encode('ascii'))

    def read(self, seconds, stop=None):
        deadline = time.monotonic()+seconds
        while True:
            if b'\n' in self.buffer:
                line, _, tail = self.buffer.partition(b'\n')
                self.buffer = bytearray(tail)
                if len(line) > 4096:
                    raise ValueError('MESSAGE_TOO_LONG')
                return line.decode('ascii').rstrip('\r')
            if len(self.buffer) > 4096:
                raise ValueError('MESSAGE_TOO_LONG')
            if stop is not None and stop.is_set():
                raise StopAfterCycle()
            left = deadline-time.monotonic()
            if left <= 0:
                raise TimeoutError('MESSAGE_DEADLINE')
            self.conn.settimeout(min(left, .2))
            try:
                data = self.conn.recv(4096)
            except socket.timeout:
                continue
            if not data:
                raise EOFError('DISCONNECTED')
            self.buffer.extend(data)


class Journal:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, kind, **values):
        record = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), kind=kind, **values)
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(_json_safe(record), ensure_ascii=False, allow_nan=False)+'\n')
            f.flush()
            os.fsync(f.fileno())


class Session:
    def __init__(self, cfg, policy, contract, watched, vision, journal, stop, emit=lambda *args: None):
        self.cfg, self.policy, self.contract, self.watched = cfg, policy, contract, watched
        self.vision, self.journal, self.stop, self.emit = vision, journal, stop, emit
        self.identifier = uuid.uuid4().hex
        self.completed = 0
        self.phase = 'WAIT_HELLO'
        self.inflight = None
        self.go_uncertain = False
        self.seen_frames = set()

    def note(self, kind, **values):
        self.journal.append(kind, session=self.identifier, completed=self.completed, phase=self.phase, **values)
        self.emit('EVENT', dict(kind=kind, phase=self.phase, completed=self.completed, **values))

    def _expect(self, io, expected, timeout=None):
        line = io.read(timeout or self.policy['stage_timeout_seconds'])
        if line.startswith('ERROR_CONT_V1|'):
            raise RuntimeError('ROBOT_ERROR:'+line)
        if line != expected:
            raise ValueError('UNEXPECTED_MESSAGE:'+line)

    def _end(self, io, seq, reason):
        io.send('END_CONT_V1|%s|%s|%s'%(self.identifier, seq, reason))
        self.note('ended_at_wait', reason=reason)
        return reason

    def serve(self, conn):
        io = Lines(conn)
        try:
            self._expect(io, 'HELLO_CONT_V1|'+digest(self.contract)+'|'+self.contract['tcp_name'], 15)
            io.send('WELCOME_CONT_V1|'+self.identifier)
            seq = 0
            while True:
                seq += 1
                self.phase = 'WAIT_READY'
                try:
                    line = io.read(self.policy['stage_timeout_seconds'], self.stop)
                except StopAfterCycle:
                    return self._end(io, seq, 'OPERATOR_STOP')
                except EOFError:
                    if self.completed:
                        self.note('pendant_ended_at_wait')
                        return 'PENDANT_FINISHED_AT_WAIT'
                    raise
                if line.startswith('ERROR_CONT_V1|'):
                    raise RuntimeError('ROBOT_ERROR:'+line)
                if line != 'READY_CONT_V1|%s|%s'%(self.identifier, seq):
                    raise ValueError('WRONG_READY_SEQUENCE:'+line)
                ready_time = datetime.now(timezone.utc)
                self.note('robot_ready_at_wait', sequence=seq)
                if self.stop.is_set():
                    return self._end(io, seq, 'OPERATOR_STOP')
                check_files(self.watched)
                self.phase = 'CAPTURE_AT_WAIT'
                job = None
                for attempt in range(self.policy['no_target_retries']+1):
                    io.send('WAIT_CONT_V1|%s|%s'%(self.identifier, seq))
                    if self.stop.wait(self.policy['settle_seconds']):
                        return self._end(io, seq, 'OPERATOR_STOP')
                    self.emit('STATUS', 'Robot tại P_WAIT — đang chụp và chọn vật mới')
                    frame, picking = self.vision.capture()
                    if self.stop.is_set():
                        self.note('capture_discarded_after_stop')
                        return self._end(io, seq, 'OPERATOR_STOP')
                    check_files(self.watched)
                    frame_key = frame.frame_id
                    if frame_key in self.seen_frames:
                        raise ValueError('REUSED_CAMERA_FRAME')
                    self.seen_frames.add(frame_key)
                    if 'REFERENCE_IDENTITY_UNVERIFIED' in picking.get('reasons', []):
                        differences = picking.get('reference', {}).get('identity_differences', {})
                        self.note('reference_identity_mismatch', differences=differences)
                        detail = '; '.join('%s: saved=%s, current=%s'%(k,v['saved'],v['current'])
                                           for k,v in differences.items())
                        self._end(io, seq, 'REFERENCE_IDENTITY_UNVERIFIED')
                        raise ValueError('REFERENCE_IDENTITY_UNVERIFIED: '+(detail or 'missing/unbound background identity'))
                    job = make_job(frame, picking, self.cfg, self.contract, ready_time)
                    if job is not None:
                        break
                    self.note('no_target', sequence=seq, attempt=attempt+1,
                              picking_status=picking.get('status'), reasons=picking.get('reasons', []))
                if job is None:
                    return self._end(io, seq, 'NO_TARGET')
                token = uuid.uuid4().hex
                job.update(job_id=token, sequence=seq, session=self.identifier)
                self.inflight = job
                prefix = '%s|%s|%s'%(self.identifier, seq, token)
                numbers = job['pregrasp']+job['grasp']+job['lift']
                self.phase = 'WAIT_ARM'
                self.note('prepare_job', job=job,
                          capture_age_ms=1000*timestamp_age(job['captured_at_utc']))
                arm_started = time.monotonic()
                io.send('PREPARE_CONT_V1|'+prefix+'|'+'|'.join('%.6f'%v for v in numbers))
                self._expect(io, 'ARM_CONT_V1|'+prefix, 15)
                arm_roundtrip_ms = 1000*(time.monotonic()-arm_started)
                check_files(self.watched)
                if self.stop.is_set():
                    self.inflight = None
                    return self._end(io, seq, 'OPERATOR_STOP')
                self.note('arm_received', arm_roundtrip_ms=arm_roundtrip_ms,
                          capture_age_ms=1000*timestamp_age(job['captured_at_utc']),
                          max_frame_age_ms=1000*self.contract['max_frame_age_seconds'])
                self.note('go_intent', job_id=token, sequence=seq)
                # The log uses fsync. Check AFTER all validation/log writes so a
                # slow disk cannot make a previously fresh image expire unnoticed.
                if self.stop.is_set():
                    self.inflight = None
                    return self._end(io, seq, 'OPERATOR_STOP')
                age = timestamp_age(job['captured_at_utc'])
                if age > self.contract['max_frame_age_seconds']:
                    self.note('capture_expired_before_go', capture_age_ms=1000*age,
                              max_frame_age_ms=1000*self.contract['max_frame_age_seconds'],
                              arm_roundtrip_ms=arm_roundtrip_ms)
                    self._end(io, seq, 'CAPTURE_EXPIRED_BEFORE_GO')
                    self.inflight = None
                    raise ValueError('CAPTURE_EXPIRED_BEFORE_GO')
                self.phase = 'SENDING_GO'
                # Set BEFORE sendall: a partial/failed write may still have reached the robot.
                self.go_uncertain = True
                io.send('GO_CONT_V1|'+prefix)
                self.note('go_sent', job_id=token, sequence=seq)
                for step in STEPS:
                    self.phase = 'WAIT_CHECK_'+step
                    self._expect(io, 'CHECK_CONT_V1|'+prefix+'|'+step)
                    self.note('permit_intent', job_id=token, step=step)
                    io.send('PERMIT_CONT_V1|'+prefix+'|'+step)
                    self.phase = 'MOVING_'+step
                    self._expect(io, 'REACHED_CONT_V1|'+prefix+'|'+step)
                    self.note('step_complete', job_id=token, step=step)
                self.phase = 'WAIT_DONE'
                self._expect(io, 'DONE_CONT_V1|'+prefix)
                self.completed += 1
                self.go_uncertain = False
                self.inflight = None
                self.note('cycle_done', job_id=token, sequence=seq)
                end = self.stop.is_set() or (self.policy['max_cycles'] > 0 and self.completed >= self.policy['max_cycles'])
                io.send('ACK_DONE_CONT_V1|'+prefix+('|'+'END' if end else '|'+'MORE'))
                if end:
                    return 'OPERATOR_STOP' if self.stop.is_set() else 'CYCLE_LIMIT_REACHED'
        except BaseException as exc:
            self.note('fault', detail=str(exc), motion_outcome_unconfirmed=self.go_uncertain,
                      job_id=self.inflight.get('job_id') if self.inflight else None)
            raise


class RunService:
    """One explicit Start -> one connection/session. Never reconnect or replay a failed job."""
    def __init__(self, cfg, policy, contract, watched, emit=lambda *args: None, vision_factory=None):
        self.cfg, self.policy, self.contract, self.watched = cfg, policy, contract, watched
        self.emit, self.stop_event = emit, threading.Event()
        self.vision_factory = vision_factory
        self.output = project_path(policy['output_dir'], cfg)/utc_stem('run')

    def request_stop(self):
        self.stop_event.set()

    def run(self):
        vision, session = None, None
        try:
            from .vision import LiveVision
            self.output.mkdir(parents=True, exist_ok=True)
            journal = Journal(self.output/'events.jsonl')
            (self.output/'contract.json').write_text(json.dumps(self.contract, indent=2, ensure_ascii=False), encoding='utf-8')
            (self.output/'configuration.json').write_text(json.dumps(_json_safe({'vision': self.cfg, 'continuous': self.policy}), indent=2, ensure_ascii=False), encoding='utf-8')
            vision = (self.vision_factory or LiveVision)(self.cfg, self.policy, self.output, self.emit)
            session = Session(self.cfg, self.policy, self.contract, self.watched, vision, journal, self.stop_event, self.emit)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind((self.policy['listen'], self.policy['port']))
                listener.listen(1)
                listener.settimeout(.2)
                self.emit('STATUS', 'Đang chờ Task Writer tại %s:%s — camera chưa kết nối'%(self.policy['listen'], self.policy['port']))
                while not self.stop_event.is_set():
                    try:
                        conn, addr = listener.accept()
                    except socket.timeout:
                        continue
                    with conn:
                        if addr[0] != self.policy['peer']:
                            journal.append('rejected_peer', address=addr[0])
                            continue
                        reason = session.serve(conn)
                        self.emit('FINISHED', dict(reason=reason, completed=session.completed, output=str(self.output)))
                        return
                self.emit('FINISHED', dict(reason='STOPPED_BEFORE_CONNECTION', completed=0, output=str(self.output)))
        except BaseException as exc:
            self.emit('FAULT', dict(detail=str(exc), phase=session.phase if session else 'INITIALIZATION',
                                   completed=session.completed if session else 0, output=str(self.output),
                                   motion_outcome_unconfirmed=session.go_uncertain if session else False))
        finally:
            try:
                if vision is not None:
                    vision.close()
            except Exception as exc:
                self.emit('CAMERA_CLOSE_ERROR', str(exc))
            self.emit('RELEASED', 'Đã đóng kết nối chương trình; có thể mở lại camera trong PhoXi Control')
