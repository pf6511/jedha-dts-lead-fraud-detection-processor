import logging


def main() -> None:
    # Imported here, not at module level: importing the package (e.g. a
    # test of payment_preparation) must not load secrets.env.
    from fraud_detection_processor.worker import run_worker

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    run_worker()


if __name__ == "__main__":
    main()
