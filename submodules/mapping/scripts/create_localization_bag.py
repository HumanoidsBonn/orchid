#!/usr/bin/env python3
"""
Create a small localization-only rosbag from the original bag.

Keeps:
  /scan
  /tf
  /tf_static
  /omni_base_controller/wheel_odom

Removes only the recorded:
  map -> odom

Usage:
  python3 create_localization_bag.py \
      /bag/playable_bag \
      /bag/localization_clean
"""

import argparse
import os
import shutil

import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from tf2_msgs.msg import TFMessage


KEEP_TOPICS = {
    "/scan",
    "/tf",
    "/tf_static",
    "/omni_base_controller/wheel_odom",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_bag",
        nargs="?",
        default="/bag/playable_bag",
        help="Input rosbag directory",
    )
    parser.add_argument(
        "output_bag",
        nargs="?",
        default="/bag/localization_clean",
        help="Output rosbag directory",
    )
    args = parser.parse_args()

    input_bag = args.input_bag
    output_bag = args.output_bag

    if not os.path.isdir(input_bag):
        raise SystemExit(f"ERROR: input bag does not exist: {input_bag}")

    if os.path.exists(output_bag):
        print(f"Removing existing output: {output_bag}")
        shutil.rmtree(output_bag)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=input_bag, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    all_topics = reader.get_all_topics_and_types()
    topic_info = {
        t.name: t
        for t in all_topics
        if t.name in KEEP_TOPICS
    }

    missing = sorted(KEEP_TOPICS - set(topic_info))
    if missing:
        print("WARNING: requested topics not found:")
        for name in missing:
            print(f"  {name}")

    print("Keeping topics:")
    for name in sorted(topic_info):
        print(f"  {name}")

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=output_bag, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )

    for name, info in topic_info.items():
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                name=name,
                type=info.type,
                serialization_format=info.serialization_format or "cdr",
                offered_qos_profiles=info.offered_qos_profiles,
            )
        )

    reader.set_filter(
        rosbag2_py.StorageFilter(
            topics=list(topic_info.keys())
        )
    )

    written_messages = 0
    removed_map_odom = 0

    while reader.has_next():
        topic, data, timestamp = reader.read_next()

        if topic == "/tf":
            msg = deserialize_message(data, TFMessage)
            filtered = TFMessage()

            for tr in msg.transforms:
                parent = tr.header.frame_id.lstrip("/")
                child = tr.child_frame_id.lstrip("/")

                if parent == "map" and child == "odom":
                    removed_map_odom += 1
                    continue

                filtered.transforms.append(tr)

            # Do not write empty TFMessages.
            if not filtered.transforms:
                continue

            data = serialize_message(filtered)

        writer.write(topic, data, timestamp)
        written_messages += 1

    print()
    print("========================================")
    print("Clean localization bag created")
    print("========================================")
    print(f"Input:  {input_bag}")
    print(f"Output: {output_bag}")
    print(f"Messages written: {written_messages}")
    print(f"map -> odom transforms removed: {removed_map_odom}")


if __name__ == "__main__":
    main()
