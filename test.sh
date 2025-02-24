export CUDA_VISIBLE_DEVICES=0
## instance level

# accelerate launch test.py --config config/class_level/man2spider.yaml
# accelerate launch test.py --config config/instance_level/running_3cls_iron_spider.yaml
# accelerate launch test.py --config config/part_level/run_spider_polar_sunglass.yaml

# accelerate launch test.py --config config/instance_level/2_monkeys/monkeys_2cls_teddy_bear_golden_retriever.yaml
# accelerate launch test.py --config config/instance_level/2_monkeys/monkeys_2cls_teddy_bear_koala.yaml

# accelerate launch test.py --config config/instance_level/badminton/badminton_2cls_wonder_woman_spiderman.yaml

# accelerate launch test.py --config config/instance_level/soap-box/soap-box.yaml

# accelerate launch test.py --config config/instance_level/2_cats/2cats_3cls_samoyed_vs_tiger_sunrise.yaml
# accelerate launch test.py --config config/instance_level/2_cats/2cats_4cls_panda_vs_poddle_bg_meadow_night.yaml

# accelerate launch test.py --config config/instance_level/2_cars/2cars_left_firetruck_right_school_bus_preserve_bg.yaml
# accelerate launch test.py --config config/instance_level/2_cars/2cars_left_firetruck_right_school_bus.yaml


## part level
# accelerate launch test.py --config config/part_level/adding_new_object/boxer-punching/thor_in_sunglasses.yaml

accelerate launch test.py --config config/part_level/adding_new_object/man_text_message/superman.yaml
accelerate launch test.py --config config/part_level/adding_new_object/man_text_message/superman+cap.yaml

# accelerate launch test.py --config config/part_level/adding_new_object/spin-ball/superman_spin_moon.yaml
# accelerate launch test.py --config config/part_level/adding_new_object/spin-ball/superman+sunglasses.yaml

# accelerate launch test.py --config config/part_level/part_level_modification/cat_flower/ginger_body.yaml
# accelerate launch test.py --config config/part_level/part_level_modification/cat_flower/ginger_head.yaml

# accelerate launch test.py --config config/part_level/part_level_modification/man_text_message/black_suit.yaml
# accelerate launch test.py --config config/part_level/part_level_modification/man_text_message/blue_shirt.yaml

# accelerate launch test.py --config config/instance_level/soely_edit/only_left.yaml
# accelerate launch test.py --config config/instance_level/soely_edit/only_right.yaml
# accelerate launch test.py --config config/instance_level/soely_edit/joint_edit.yaml

# ## class level
# accelerate launch test.py --config config/class_level/car/posche.yaml

# accelerate launch test.py --config config/class_level/tennis/1cls_man2iron_man.yaml
# accelerate launch test.py --config config/class_level/tennis/3cls_batman_snow-court_iced-wall.yaml

# accelerate launch test.py --config config/class_level/wolf/wolf_to_pig.yaml