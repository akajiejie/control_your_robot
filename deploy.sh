python example/deploy/deploy.py \
    --model_name "test_policy"\
    --model_class "TestModel"\
    --model_path "path/to/ckpt"\
    --task_name "test"\
    --robot_name "test_robot"\
    --robot_class "TestRobot"\
    --overrides \
    --test "test for robot"
    # --video "cam_head"\