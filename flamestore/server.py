import _flamestore_server


class MasterServer(_flamestore_server.MasterServer):

    def __init__(self, engine, *args, **kwargs):
        super().__init__(engine.get_internal_mid(), *args, **kwargs)

