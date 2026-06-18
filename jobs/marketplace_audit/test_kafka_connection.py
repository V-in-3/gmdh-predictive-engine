# import unittest
# from confluent_kafka import Consumer
#
# class TestKafkaConnectivity(unittest.TestCase):
#     def setUp(self):
#         self.config = {
#             'bootstrap.servers': 'localhost:9092',
#             'group.id': 'unittest-group',
#             'socket.timeout.ms': 5000,
#             'broker.address.family': 'v4'  # This removes IPv6 errors we saw
#         }
#
#     def test_can_list_topics(self):
#         """Check if we can see our topics in Docker"""
#         consumer = Consumer(self.config)
#         metadata = consumer.list_topics(timeout=5)
#         topics = list(metadata.topics.keys())
#
#         self.assertIn('raw-subscriptions', topics, "Topic raw-subscriptions should exist")
#         self.assertIn('subscriptions-dlq', topics, "DLQ topic should exist")
#         consumer.close()
#
#     def test_topic_has_messages(self):
#         """Check if the topic has at least one message (our event)"""
#         from confluent_kafka import TopicPartition
#         consumer = Consumer(self.config)
#         tp = TopicPartition('raw-subscriptions', 0)
#         low, high = consumer.get_watermark_offsets(tp)
#         consumer.close()
#
#         self.assertGreaterEqual(high, 1, "There should be at least 1 message in Kafka")
#
# if __name__ == '__main__':
#     unittest.main()