import argparse
import logging
import os

import tornado.ioloop

from xorq_web.app import make_app

LOG_DIR = os.path.join(os.path.expanduser("~"), ".xorq", "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="xorq web server")
    parser.add_argument("--port", type=int, default=8456, help="Port to listen on")
    parser.add_argument(
        "--buckaroo-port", type=int, default=8455, help="Buckaroo server port"
    )
    args = parser.parse_args()

    logging.basicConfig(
        filename=os.path.join(LOG_DIR, "web_server.log"),
        level=logging.DEBUG,
        format="%(asctime)s pid=%(process)d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("xorq.web")
    log.info(
        "xorq web server starting — port=%d buckaroo_port=%d pid=%d",
        args.port,
        args.buckaroo_port,
        os.getpid(),
    )

    app = make_app(buckaroo_port=args.buckaroo_port)
    app.listen(args.port)
    log.info("xorq web server listening on http://localhost:%d", args.port)

    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
