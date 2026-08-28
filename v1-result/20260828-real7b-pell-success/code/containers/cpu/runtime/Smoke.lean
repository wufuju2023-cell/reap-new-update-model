import ReapRuntime

set_option reap.max_goals 8
set_option reap.max_steps 8
set_option reap.num_samples 2
set_option reap.num_premises 0

theorem reap_training_cpu_smoke : True := by
  reapTrainingMCTS

