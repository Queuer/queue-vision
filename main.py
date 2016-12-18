from multiprocessing import Pool

from config import config
from recognition.recognize import Recognizer

from tornado import ioloop, web, gen
from tornado.escape import json_encode
from tornado import httpclient

recognizer = None
pool = None
http_client = httpclient.AsyncHTTPClient()


def init():
    global recognizer
    print('Initializing recognizer...')
    recognizer = Recognizer()


def recognize(path):
    return recognizer.recognize(path)


# noinspection PyAbstractClass
class MainHandler(web.RequestHandler):
    @web.asynchronous
    @gen.engine
    def get(self):
        image_url = self.get_argument('image')

        image_response = yield self.load_image(image_url)
        recognition_result = yield self.recognize(image_response.body)

        self.write_json({
            'count': len(recognition_result)
        })

    def write_json(self, data):
        self.add_header('Content-Type', 'application/json')
        self.write(json_encode(data))
        self.finish()

    @staticmethod
    def load_image(url):
        request = httpclient.HTTPRequest(url)
        return http_client.fetch(request)

    @staticmethod
    def recognize(path):
        return gen.Task(
            lambda callback: pool.apply_async(
                func=recognize,
                args=(path,),
                callback=callback
            )
        )


def app():
    return web.Application([
        (r"/api/v1/recognize", MainHandler),
    ])


if __name__ == '__main__':
    pool = Pool(
        processes=config['PROCESS_POOL_SIZE'],
        initializer=init
    )

    app = app()
    app.listen(config['PORT'])

    print('Listening port {port}...'.format(port=config['PORT']))

    ioloop.IOLoop.current().start()
    pool.close()
