#!/usr/bin/env python2

from tasks import drive

import flask
from flask import request
import flask_socketio as io
import json
import numpy as np
import threading
from time import gmtime, strftime

class State:
    def __init__(self, n_players):
        self.n_players = n_players

        self.lock = threading.RLock()

        self.rooms = {}
        self.room_assignments = {}
        self.next_room = 0

        self.games = {}
        self.pending_actions = {}
        self.scores = {}
        self.traces = {}
        self.done = {}

    def join(self, client_id):
        with self.lock:
            free_rooms = [
                    (n, c) for n, c in self.rooms.items()
                    if len(c) < self.n_players]
            if len(free_rooms) == 0:
                room = self.next_room
                self.next_room += 1
                assert room not in self.rooms
                self.rooms[room] = [client_id]
            else:
                room, clients = free_rooms[0]
                clients.append(client_id)
            self.room_assignments[client_id] = room
            return room, len(self.rooms[room]) == self.n_players

    def initialize(self, room, game):
        with self.lock:
            self.games[room] = game
            self.pending_actions[room] = [None] * self.n_players
            self.scores[room] = 0
            self.traces[room] = []
            self.done[room] = False

    def log(self, room, trace_repr):
        with self.lock:
            self.traces[room].append(trace_repr)

    def save_log(self, room, token):
        try:
            timestamp = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
            path = "server_logs/" + timestamp + "_" + token + ".json"
            with open(path, "w") as log_f:
                json.dump(self.traces[room], log_f)
        except Exception as e:
            print e

    def close(self, room):
        with self.lock:
            clients = self.rooms[room]
            del self.rooms[room]
            for client in clients:
                del self.room_assignments[client]

app = flask.Flask(__name__)
app.config["SECRET_KEY"] = "secret"
socketio = io.SocketIO(app)
state = State(2)

@app.route("/")
def index():
    return flask.render_template("index.html")

@socketio.on("connect")
def connect():
    room, ready = state.join(request.sid)
    io.emit("join", {"room": room})
    io.join_room(room)
    if ready:
        game = initialize()
        state.initialize(room, game)
        state.log(room, trace_repr(game))
        for i_client, client in enumerate(state.rooms[room]):
            io.emit("begin", init_repr(game, i_client), room=client)

@socketio.on("disconnect")
def disconnect():
    if request.sid not in state.room_assignments:
        # this session was already terminated by another client
        return
    room = state.room_assignments[request.sid]
    io.emit("end", {"reason": "disconnect"}, room=room)
    state.close(room)
    io.close_room(room)

@socketio.on("action")
def action(data):
    with state.lock:
        room = state.room_assignments[request.sid]
        if state.done[room]:
            return
        i_player = state.rooms[room].index(request.sid)
        pending_actions = state.pending_actions[room]
        pending_actions[i_player] = data
        if None not in pending_actions:
            state.log(room, pending_actions)
            ngame, reward, stop = step(state.games[room], pending_actions)
            state.log(room, trace_repr(ngame))
            state.games[room] = ngame
            state.scores[room] += reward
            for i_client, client in enumerate(state.rooms[room]):
                io.emit("step", state_repr(state.games[room], i_client), room=client)
            if stop:
                state.done[room] = True
                token = "".join([chr(ord("a") + np.random.randint(16)) for _ in range(8)])
                state.save_log(room, token)
                io.emit("end", 
                        {"reason": "game", 
                            "score": "%0.1f" % state.scores[room],
                            "token": token},
                        room=room)
            state.pending_actions[room] = [None] * state.n_players

#-------------------------------------------------------------------------------

task = drive.DriveTask()

def initialize():
    inst = task.get_instance(None)
    return (inst, ([], []))

def init_repr(game, agent):
    game_state, chats = game
    car = game_state.cars[agent]
    return {
        "road": game_state.road.tolist(),
        "pos": car.pos,
        "dir": car.dir,
        "goal": car.goal,
        "done": car.done
    }

def state_repr(game, agent):
    game_state, chats = game
    car = game_state.cars[agent]
    return {
        "pos": car.pos,
        "dir": car.dir,
        "goal": car.goal,
        "done": car.done,
        "chat": chats[agent]
    }

def trace_repr(game):
    game_state, chats = game
    cars = []
    for car in game_state.cars:
        cars.append({
            "pos": car.pos,
            "dir": car.dir,
            "goal": car.goal,
            "done": car.done
        })
    return {
        "road": game_state.road.tolist(),
        "cars": cars
    }

def step(game, actions):
    game_state, chats = game
    newchats = tuple(list(c) for c in chats)
    game_actions = [a["action"] for a in actions]
    for player in range(len(actions)):
        msg = actions[player]["message"]
        if msg != "":
            newchats[player].append("You said: " + msg)
        for oplayer in range(len(actions)):
            if oplayer == player:
                continue
            omsg = actions[oplayer]["message"]
            if omsg != "":
                newchats[player].append("Partner said: " + omsg)
    for c in newchats:
        c.append("<br><hr>")
    ngame_state, reward, stop = game_state.step(game_actions)
    return (ngame_state, newchats), reward, stop

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    socketio.run(app, debug=True, host="fromage.banatao.berkeley.edu", port=5000)
