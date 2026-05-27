import sys
sys.path.insert(0, ".")

from src.document_processing.kafka_consumer import DocumentKafkaConsumer


def main():
    consumer = DocumentKafkaConsumer()
    print("Kafka consumer started. Waiting for messages...")
    try:
        consumer.consume()
    except KeyboardInterrupt:
        print("\nConsumer stopped.")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
