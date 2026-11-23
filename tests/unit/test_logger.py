from src.core.logger import CustomLogger


def test_custom_logger_supports_percent_style_args():
    messages = []
    logger = CustomLogger(console=False, sinks=[lambda level, message: messages.append(message)])

    logger.info("Robot %s ready with %d action(s)", "hsrb_bot", 12)

    assert messages == ["Robot hsrb_bot ready with 12 action(s)"]


def test_custom_logger_keeps_print_style_fallback():
    messages = []
    logger = CustomLogger(console=False, sinks=[lambda level, message: messages.append(message)])

    logger.info("end_effector_frame", "hand_palm_link")

    assert messages == ["end_effector_frame hand_palm_link"]
