import socketserver

class MyTCPHandler(socketserver.BaseRequestHandler):
    """
    The request handler used by the server.
    Instantiated once per connection to the server, and must
    """

    def handle(self):
        # self.request is the TCP socket connected to the client
        self.data = self.request.recv(1024).strip()
        print("{} wrote:".format(self.client_address[0]))
        print(self.data)
        # just send back the same data, but upper-cased
        self.request.sendall(self.data.upper())

if __name__ == "__main__":
    HOST, PORT = "192.168.0.6", 1024
    server = socketserver.TCPServer((HOST, PORT), MyTCPHandler)
    # will keep running until you interrupt the program with Ctrl-C
    server.serve_forever()