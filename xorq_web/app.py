import os

import tornado.web

from xorq_web.handlers import (
    CatalogIndexHandler,
    DiscardHandler,
    ExpressionDetailHandler,
    HealthHandler,
    PromoteHandler,
    RunsHandler,
    SessionExpressionDetailHandler,
)


def make_app(buckaroo_port: int = 8455) -> tornado.web.Application:
    return tornado.web.Application(
        [
            (r"/", CatalogIndexHandler),
            (r"/runs", RunsHandler),
            # Session routes must come before /entry/ to avoid ambiguity
            (r"/session/(.+)/promote", PromoteHandler),
            (r"/session/(.+)/discard", DiscardHandler),
            (r"/session/(.+)", SessionExpressionDetailHandler),
            (r"/entry/(.+)", ExpressionDetailHandler),
            (r"/health", HealthHandler),
        ],
        template_path=os.path.join(os.path.dirname(__file__), "templates"),
        static_path=os.path.join(os.path.dirname(__file__), "static"),
        buckaroo_port=buckaroo_port,
    )
