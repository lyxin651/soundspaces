# ClassDOA V1 R3C-A.2 Regression Test Closure

## Scope

This closure addresses the previously observed planner-test hang only. No R3C-B work, Pilot005 PLAN/root creation, bulk repair, SoundSpaces render, WAV/RIR generation, Pilot004 modification, Step4G/finalize, or training was performed.

The production generation-code identity remains:

`PILOT005_GENERATION_CODE_COMMIT = 732d960844ebe70d0bf15ad0dc8da2ea3deeef02`

The repair integration code is unchanged by this closure. The only code after that identity is test-only hardening in commit `b2cfb959f72b116eeeb596989e35a82cf27cf337`, followed by this evidence commit.

## Hang isolation and comparison

The isolated command was:

```text
/home/leiyuxin/miniconda3/envs/ss/bin/python -m unittest tests.tools.test_clsdoa_v1_step3_r13a.Pilot004CommandSurfaceTests.test_plan_rejects_dirty_repo_before_creating_root_and_nonempty_root -v
```

On `732d960` (the branch HEAD at the time of the first reproduction), the command was run with an external `timeout 30s`. It timed out with exit code `124`; the parent unittest process was observed in `do_sys_poll` and its child was:

```text
/home/leiyuxin/miniconda3/envs/ss/bin/python tools/clsdoa_v1/pilot_dataset.py plan --config configs/active_audition/clsdoa_v1_pilot_004.yaml --root /tmp/tmplsefg5ou/new-root
```

The earlier 180-second run showed the same parent/child pattern and was cleaned up. The exact test was then run against comparison commit `e29a0e5e9ee58a7cbea29234e527b8a704779cb7` in a detached temporary worktree under the same environment; it also timed out at the external 30-second limit. This establishes an existing test/environment issue rather than a regression introduced by `732d960`.

The test harness was minimally corrected in `tests/tools/test_clsdoa_v1_step3_r13a.py` (commit `b2cfb95`): it now creates a temporary dirty marker inside the repository for both CLI invocations, uses a bounded subprocess timeout, and removes the marker in `finally`. The planner therefore reaches the intended dirty-HEAD rejection gate instead of launching the real geometry planner on a clean worktree. The test semantics remain unchanged: the new root is not created and a pre-existing non-empty root remains untouched.

## Final regression results

The isolated dirty-repo test passed in both modes:

```text
python -m unittest ...test_plan_rejects_dirty_repo_before_creating_root_and_nonempty_root -v: PASS
python -O -m unittest ...test_plan_rejects_dirty_repo_before_creating_root_and_nonempty_root -v: PASS
```

The focused repair suite also passed in both modes:

```text
python -m unittest tests.tools.test_clsdoa_v1_r3c_repair -v: 11/11 PASS
python -O -m unittest tests.tools.test_clsdoa_v1_r3c_repair -v: 11/11 PASS
```

Complete discovery then passed without failures or errors:

```text
python -m unittest discover -s tests/tools -p 'test_*.py' -v: 100/100 PASS
python -m unittest discover -s tests -p 'test_*.py' -v: 206/206 PASS
```

No related child process remained after the runs. `git diff --check` passed and the worktree was clean before this evidence file was committed.

## Frozen-state checks

The canonical Pilot004 worktree remains detached at `5388d17ef18919a7aa7cd911b6b239f1d91813f1` and clean. Pilot005 does not exist in the repair worktree or canonical Pilot004 worktree. Step2 frozen inputs, production renderer, FOA adapter, and Pilot001/002/003 were not modified.

## Verdict

`STEP 4F-R3C-A.2 REGRESSION TEST CLOSURE PASS — PENDING HUMAN REVIEW`

The next action is human review only. R3C-B and subsequent Pilot005 generation remain stopped.
