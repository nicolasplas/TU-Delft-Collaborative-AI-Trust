import json
import csv
import pandas as pd
import math
from typing import final, List, Dict, Final
import enum, random
from bw4t.BW4TBrain import BW4TBrain
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import GrabObject, DropObject
from matrx.messages.message import Message
from matrx.actions.action import Action

Trust_Level = 0.75
delete_age = 8


def findRoom(location, state):
    room = None
    rooms = state.get_with_property({'room_name'})
    for item in rooms:
        if item['location'] == location:
            room = item['room_name'].split('_')[1]
            break

    return room


class Phase(enum.Enum):
    PLAN_PATH_TO_ROOM = 1,
    FOLLOW_PATH_TO_ROOM = 2,
    OPEN_DOOR = 3,
    WAIT_FOR_DOOR = 4.
    ENTERING_ROOM = 5,
    SEARCHING_ROOM = 6,
    FOLLOW_PATH_TO_DROP = 7,
    DROP_OBJECT = 8,
    CHECK_GOALS = 9,
    FOLLOW_PATH_TO_GOAL = 10,
    PICK_UP_GOAL_BLOCK = 11,
    PUT_AWAY_WRONG_BLOCK = 12,
    WAIT_FOR_FINISH = 13,
    MOVE_GOAL_BLOCK = 14,
    UPDATE_GOAL_LIST = 15,
    MOVING_TO_KNOWN_BLOCK = 16,
    RUN_BACK = 17


class BaseAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.PLAN_PATH_TO_ROOM
        self._teamMembers = []

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)
        self._goalBlocks = None
        self._goalsInitialized = False
        self._carrying = None
        self._carryingO = None
        self._goalsWrong = []
        self._checkGoals = []
        self._possibleGoalBLocks = []
        self._notExplored = []
        self._trustBeliefs = {}
        self._teamStatus = {}
        self._teamObservedStatus = {}
        self._age = 0

    def filter_observations(self, state):
        self._age += 1
        agent_name = state[self.agent_id]['obj_id']
        loc = 0
        df = pd.read_csv("agents1/agents.csv")
        if len(self._teamMembers) == 0:
            for member in state['World']['team_members']:
                if member != agent_name and member not in self._teamMembers:
                    self._teamMembers.append(member)
                    self._trustBeliefs[member] = {'rating': df.loc[loc, 'rating'], 'age': self._age}
                    loc += 1


            if self._age % 25 == 0:
                self._sendMessage('Trustbeliefs: ' + str(self._trustBeliefs), agent_name)
            closest_agents = state.get_closest_agents()
            if closest_agents is not None:
                for item in closest_agents:
                    name = item['name']
                    location = item['location']
                    is_carrying = []
                    if len(item['is_carrying']) > 0:
                        for block in item['is_carrying']:
                            block = {"size": block['visualization']['size'], "shape": block['visualization']["shape"],
                                     "colour": block['visualization']["colour"]}
                            is_carrying.append(block)
                    self._teamObservedStatus[name] = {'location': location, 'is_carrying': is_carrying,
                                                      'age': self._age}
                    self._sendMessage('status of ' + name + ': location: '
                                      + str(location) + ', is carrying: ' + str(is_carrying), agent_name)

            receivedMessages = self._processMessages(self._teamMembers)

            for member in self._teamMembers:
                if member in self._teamObservedStatus and self._teamObservedStatus[member] is not None:
                    if self._age - self._teamObservedStatus[member]['age'] > delete_age:
                        self._teamObservedStatus[member] = None

            for member in self._teamMembers:
                for message in receivedMessages[member]:
                    self._parseMessage(message, member, agent_name)

            # Update trust beliefs for team members
            self._trustBlief(agent_name, state)
            csv_file.close()

            df = pd.read_csv("agents1/agents.csv")
            loc = 0
            for member in self._teamMembers:
                df.loc[loc, 'age'] = self._age
                df.loc[loc, 'rating'] = self._trustBeliefs[member]['rating']
                loc += 1
            df.to_csv("agents1/agents.csv", index = False)
            return state

    def decide_on_bw4t_action(self, state: State):
        if not self._goalsInitialized:
            self._goalBlocks = state.get_with_property({'is_goal_block': True})
            self._goalsInitialized = True
            self._notExplored = [door for door in state.values()
                                 if 'class_inheritance' in door and 'Door' in door['class_inheritance']]

        agent_name = state[self.agent_id]['obj_id']
        # Add team members
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)

        # Check if there are objects lying around the agent.
        objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
        if objects is not None:
            for o in objects:
                count = 0
                for g in self._goalBlocks:
                    if o['location'] == g['location']:
                        count += 1
                if count == 0:
                    for g in self._goalBlocks:
                        if o['visualization']['shape'] == g['visualization']['shape'] and o['visualization'][
                            'colour'] == \
                                g['visualization']['colour'] and len(o['carried_by']) == 0:
                            if o['location'] not in self._possibleGoalBLocks:
                                self._possibleGoalBLocks.append(o['location'])
                            self._sendMessage('Found goal block {\"size\": ' + str(
                                o['visualization']['size']) + ', \"shape\": ' + str(
                                o['visualization']['shape']) + ', \"colour\": \"' + str(
                                o['visualization']['colour']) + '\"} at location ' + str(o['location']), agent_name)

        while True:
            if Phase.PLAN_PATH_TO_ROOM == self._phase:
                self._navigator.reset_full()
                rooms = [door for door in state.values()
                         if 'class_inheritance' in door and 'Door' in door['class_inheritance']]
                if len(rooms) == 0:
                    return None, {}
                # If not every room has been explored, pick one randomly from the non explored, else pick randomly pick a door
                if len(self._notExplored) != 0:
                    self._door = random.choice(self._notExplored)
                else:
                    self._door = random.choice(rooms)
                doorLoc = self._door['location']
                # Location in front of door is south from door
                doorLoc = doorLoc[0], doorLoc[1] + 1
                # Send message of current action
                self._sendMessage('Moving to ' + self._door['room_name'], agent_name)
                self._navigator.add_waypoints([doorLoc])
                self._phase = Phase.FOLLOW_PATH_TO_ROOM

            if Phase.FOLLOW_PATH_TO_ROOM == self._phase:
                self._state_tracker.update(state)
                # Follow path to room
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                # If the door is not open, open the door.
                if self._door['is_open']:
                    self._phase = Phase.ENTERING_ROOM
                else:
                    self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                self._phase = Phase.WAIT_FOR_DOOR
                # Open door
                self._sendMessage('Opening door of ' + self._door['room_name'], agent_name)
                return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}

            if Phase.WAIT_FOR_DOOR == self._phase:
                self._phase = Phase.ENTERING_ROOM
                return None, {}

            if Phase.ENTERING_ROOM == self._phase:
                if self._door in self._notExplored:
                    self._notExplored.remove(self._door)
                # Send message of action
                self._sendMessage('Searching through ' + self._door['room_name'], agent_name)
                self._navigator.reset_full()
                objects = state.get_room_objects(self._door['room_name'])
                # Add all traversable tiles as waypoints
                for o in objects:
                    if o['is_traversable']:
                        self._navigator.add_waypoints([o['location']])
                self._phase = Phase.SEARCHING_ROOM

            if Phase.SEARCHING_ROOM == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                # If there is a goal block in the room pick it up and take it to the goal
                if objects is not None:
                    for o in objects:
                        for g in self._goalBlocks:
                            if o['visualization']['shape'] == g['visualization']['shape'] and o['visualization'][
                                'colour'] == g['visualization']['colour'] and len(o['carried_by']) == 0:
                                if o['location'] in self._possibleGoalBLocks:
                                    self._possibleGoalBLocks.remove(o['location'])
                                self._sendMessage('Found goal block {\"size\": ' + str(
                                    o['visualization']['size']) + ', \"shape\": ' + str(
                                    o['visualization']['shape']) + ', \"colour\": \"' + str(
                                    o['visualization']['colour']) + '\"} at location ' + str(o['location']), agent_name)
                                self._sendMessage('Picking up goal block {\"size\": ' + str(
                                    o['visualization']['size']) + ', \"shape\": ' + str(
                                    o['visualization']['shape']) + ', \"colour\": \"' + str(
                                    o['visualization']['colour']) + '\"} at location ' + str(
                                    o['location']), agent_name)
                                self._phase = Phase.FOLLOW_PATH_TO_DROP
                                self._navigator.reset_full()
                                self._navigator.add_waypoints([g['location']])
                                action = GrabObject.__name__
                                action_kwargs = {}
                                action_kwargs['object_id'] = o['obj_id']
                                self._carrying = g
                                self._carryingO = o
                                return action, action_kwargs
                if action != None:
                    return action, {}
                # If you explored the entire room, find a new one or go to a known block
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK

            if Phase.FOLLOW_PATH_TO_DROP == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.DROP_OBJECT

            if Phase.DROP_OBJECT == self._phase:
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK

                # If there already is a block in this location, move to a different location and drop the block there.
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        if o['location'] == self.state.get_self()['location']:
                            self._navigator.reset_full()
                            self._navigator.add_waypoints([self._carryingO['location']])
                            self._phase = Phase.FOLLOW_PATH_TO_DROP
                            return None, {}

                self._goalBlocks.remove(self._carrying)
                # If there are more goal blocks to find, update your goalblock list. Else check if it is a solution
                if len(self._goalBlocks) >= 1:
                    self._phase = Phase.UPDATE_GOAL_LIST
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([self._goalBlocks[0]['location']])
                    self._checkGoals = []
                    for g in self._goalBlocks:
                        self._checkGoals.append(g)
                elif len(self._goalBlocks) == 0:
                    self._goalBlocks = state.get_with_property({'is_goal_block': True})
                    self._phase = Phase.CHECK_GOALS

                self._sendMessage('Dropped goal block {\"size\": ' + str(
                    self._carryingO['visualization']['size']) + ', \"shape\": ' + str(
                    self._carryingO['visualization']['shape']) + ', \"colour\": \"' + str(
                    self._carryingO['visualization']['colour']) + '\"} at location ' + str(
                    self.state.get_self()['location']),
                                  agent_name)

                self._carrying = None
                self._carryingO = None
                return DropObject.__name__, {}

            if Phase.CHECK_GOALS == self._phase:
                self._phase = Phase.FOLLOW_PATH_TO_GOAL
                # If there was a wrong block on a goal, try and find a goal block
                if len(self._goalsWrong) != 0:
                    if len(self._possibleGoalBLocks) == 0:
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    else:
                        block = self._possibleGoalBLocks[0]
                        self._navigator.reset_full()
                        self._navigator.add_waypoints([block])
                        self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                    return None, {}
                # If all goal blocks have been checked, try and find a goal block
                if len(self._goalBlocks) == 0:
                    if len(self._possibleGoalBLocks) == 0:
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    else:
                        block = self._possibleGoalBLocks[0]
                        self._navigator.reset_full()
                        self._navigator.add_waypoints([block])
                        self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                    return None, {}
                goal = self._goalBlocks[0]
                self._navigator.reset_full()
                self._navigator.add_waypoints([goal['location']])

            if Phase.FOLLOW_PATH_TO_GOAL == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.PICK_UP_GOAL_BLOCK

            if Phase.PICK_UP_GOAL_BLOCK == self._phase:
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        if o['location'] == self.state.get_self()['location']:
                            # If the current goal location has the wrong block on it, remove it
                            if o['visualization']['shape'] != self._goalBlocks[0]['visualization']['shape'] or \
                                    o['visualization']['colour'] != self._goalBlocks[0]['visualization']['colour']:
                                self._phase = Phase.PUT_AWAY_WRONG_BLOCK
                                self._navigator.reset_full()
                                self._navigator.add_waypoints(
                                    [[self._goalBlocks[0]['location'][0], self._goalBlocks[0]['location'][1] - 3]])
                            # Pick up the goal block that is correct and then immediately drop it
                            else:
                                self._carryingO = o
                                self._phase = Phase.MOVE_GOAL_BLOCK
                            self._sendMessage('Picking up goal block {\"size\": ' + str(
                                o['visualization']['size']) + ', \"shape\": ' + str(
                                o['visualization']['shape']) + ', \"colour\": \"' + str(
                                o['visualization']['colour']) + '\"} at location ' + str(
                                o['location']), agent_name)
                            if o['location'] in self._possibleGoalBLocks:
                                self._possibleGoalBLocks.remove(o['location'])
                            action = GrabObject.__name__
                            self._carryingO = o
                            action_kwargs = {}
                            action_kwargs['object_id'] = o['obj_id']
                            return action, action_kwargs
                # Find a room or block
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK

            if Phase.PUT_AWAY_WRONG_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._goalsWrong.append(self._goalBlocks[0])

                self._phase = Phase.CHECK_GOALS

                return DropObject.__name__, {}

            if Phase.MOVE_GOAL_BLOCK == self._phase:
                self._phase = Phase.CHECK_GOALS
                self._goalBlocks.remove(self._goalBlocks[0])

                self._sendMessage('Dropped goal block {\"size\": ' + str(
                    self._carryingO['visualization']['size']) + ', \"shape\": ' + str(
                    self._carryingO['visualization']['shape']) + ', \"colour\": \"' + str(
                    self._carryingO['visualization']['colour']) + '\"} at location ' + str(
                    self.state.get_self()['location']),
                                  agent_name)
                self._carryingO = None
                return DropObject.__name__, {}

            if Phase.UPDATE_GOAL_LIST == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                if len(self._goalBlocks) == 0:
                    self._goalBlocks = state.get_with_property({'is_goal_block': True})
                    self._phase = Phase.CHECK_GOALS
                    return None, {}
                # If there is a block on the goal, update the goallist
                if len(self._checkGoals) == 0:
                    if len(self._possibleGoalBLocks) == 0:
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    else:
                        block = self._possibleGoalBLocks[0]
                        self._navigator.reset_full()
                        self._navigator.add_waypoints([block])
                        self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                    return None, {}
                goal = self._checkGoals[0]
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        if o['location'] == goal['location']:
                            self._goalBlocks.remove(goal)
                            if len(self._goalBlocks) == 0:
                                self._goalBlocks = state.get_with_property({'is_goal_block': True})
                                self._phase = Phase.CHECK_GOALS
                                return None, {}
                            elif len(self._checkGoals) == 0:
                                if len(self._possibleGoalBLocks) == 0:
                                    self._phase = Phase.PLAN_PATH_TO_ROOM
                                else:
                                    block = self._possibleGoalBLocks[0]
                                    self._navigator.reset_full()
                                    self._navigator.add_waypoints([block])
                                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                            else:
                                next = self._checkGoals[0]
                                self._navigator.reset_full()
                                self._navigator.add_waypoints([next['location']])
                self._checkGoals.remove(goal)

            if Phase.MOVING_TO_KNOWN_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                # If the known block is a goal block, pick it up and bring it to the goal
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        if o['location'] == self._possibleGoalBLocks[0]:
                            for g in self._goalBlocks:
                                if o['visualization']['shape'] == g['visualization']['shape'] and o['visualization'][
                                    'colour'] == g['visualization']['colour']:
                                    self._sendMessage('Found goal block {\"size\": ' + str(
                                        o['visualization']['size']) + ', \"shape\": ' + str(
                                        o['visualization']['shape']) + ', \"colour\": \"' + str(
                                        o['visualization']['colour']) + '\"} at location ' + str(o['location']),
                                                      agent_name)
                                    self._sendMessage('Picking up goal block {\"size\": ' + str(
                                        o['visualization']['size']) + ', \"shape\": ' + str(
                                        o['visualization']['shape']) + ', \"colour\": \"' + str(
                                        o['visualization']['colour']) + '\"} at location ' + str(
                                        o['location']), agent_name)
                                    self._phase = Phase.FOLLOW_PATH_TO_DROP
                                    self._navigator.reset_full()
                                    self._navigator.add_waypoints([g['location']])
                                    action = GrabObject.__name__
                                    action_kwargs = {}
                                    action_kwargs['object_id'] = o['obj_id']
                                    self._carrying = g
                                    self._carryingO = o
                                    return action, action_kwargs
                self._possibleGoalBLocks.remove(self._possibleGoalBLocks[0])
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK

    def _sendMessage(self, mssg, sender):
        '''
        Enable sending messages in one line of code
        '''
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages:
            self.send_message(msg)

    def _processMessages(self, teamMembers):
        '''
        Process incoming messages and create a dictionary with received messages from each team member.
        '''
        receivedMessages = {}
        for member in teamMembers:
            receivedMessages[member] = []
        for mssg in self.received_messages:
            for member in teamMembers:
                if mssg.from_id == member:
                    receivedMessages[member].append(mssg.content)
        return receivedMessages

    def _trustBlief(self, name, state):
        '''
        Baseline implementation of a trust belief. Creates a dictionary with trust belief scores for each team member, for example based on the received messages.
        '''

        # You can change the default value to your preference

        theta = 0.17
        mu = 0.5
        increment = 0.01

        for member in self._teamMembers:
            if member not in self._teamStatus or member not in self._teamObservedStatus:
                continue
            self._trustBeliefs[member]['age'] = self._age
            rating = self._trustBeliefs[member]['rating']
            if self._teamObservedStatus[member] is not None and self._teamStatus[member]['action'] == 'searching':
                if self._teamObservedStatus[member] is not None:
                    if findRoom(self._teamObservedStatus[member]['location'], state) == self._teamStatus[member][
                        'room']:
                        rating += \
                            increment * (1 / (theta * math.sqrt(2 * math.pi)) *
                                         math.exp(-0.5 * math.pow((rating - mu) / theta, 2)))

            elif self._teamStatus[member]['action'] == 'carrying':
                if self._teamObservedStatus[member] is not None and len(
                        self._teamObservedStatus[member]['is_carrying']) > 0:
                    if self._teamStatus[member]['block'] in self._teamObservedStatus[member]['is_carrying']:
                        rating += \
                            increment * (1 / (theta * math.sqrt(2 * math.pi)) *
                                         math.exp(-0.5 * math.pow((rating - mu) / theta, 2)))

                    else:
                        rating -= \
                            10 * increment * (1 / (theta * math.sqrt(2 * math.pi)) *
                                             math.exp(-0.5 * math.pow((rating - mu) / theta, 2)))
            if rating < 0:
                rating = 0
            if rating > 1:
                rating = 1
            self._trustBeliefs[member]['rating'] = rating

    def _parseMessage(self, message, member, myself):
        string_list = message.split(" ")
        if string_list[0] == "Opening" and string_list[1] == "door":
            room_number = string_list[3].split("_")[1]
            self._teamStatus[member] = {'action': 'opening', 'room': room_number, 'age': self._age}
        if string_list[0] == "Searching" and string_list[1] == "through":
            room_number = string_list[2].split("_")[1]
            self._teamStatus[member] = {'action': 'searching', 'room': room_number, 'age': self._age}
        if string_list[0] == "Found" and string_list[1] == "goal":
            block = message.split('{')[1]
            block = '{' + block.split('}')[0] + '}'
            block = block.replace("'", '"')
            block = block.replace("True", "true")
            block = block.replace("False", "false")
            block = json.loads(block)
            self._teamStatus[member] = {'action': 'finding', 'block': block, 'age': self._age}
            # If the trust is high enough, add goal block to possible goal blocks
            if self._trustBeliefs[member]['rating'] >= Trust_Level and block['colour'] != "" and member != self.agent_name:
                location = message.split('(')[1].split(')')[0].split(', ')
                tupl = (int(location[0]), int(location[1]))
                isgoal = False
                for g in self.state.get_with_property({'is_goal_block': True}):
                    if tupl == g['location']:
                        isgoal = True
                if not isgoal and tupl not in self._possibleGoalBLocks:
                    self._possibleGoalBLocks.append(tupl)
        if string_list[0] == "Picking" and string_list[1] == "up":
            block = message.split('{')[1]
            block = '{' + block.split('}')[0] + '}'
            block = block.replace("'", '"')
            block = block.replace("True", "true")
            block = block.replace("False", "false")
            block = json.loads(block)
            self._teamStatus[member] = {'action': 'carrying', 'block': block, 'age': self._age}
            # If the trust is high enough, add remove goal block from possible goal blocks
            if self._trustBeliefs[member]['rating'] >= Trust_Level and member != self.agent_name:
                location = message.split('(')[1].split(')')[0].split(', ')
                tupl = (int(location[0]), int(location[1]))
                for b in self._possibleGoalBLocks:
                    if b == tupl:
                        self._possibleGoalBLocks.remove(b)
        if string_list[0] == "Dropping" and string_list[1] == "goal":
            block = message.split('{')[1]
            block = '{' + block.split('}')[0] + '}'
            block = block.replace("'", '"')
            block = block.replace("True", "true")
            block = block.replace("False", "false")
            block = json.loads(block)
            self._teamStatus[member] = {'action': 'dropping', 'block': block, 'age': self._age}
            # If the trust is high enough, remove goal block from goalblocks
            if self._trustBeliefs[member]['rating'] >= Trust_Level and member != self.agent_name:
                location = message.split('(')[1].split(')')[0].split(', ')
                tupl = (int(location[0]), int(location[1]))
                for g in self._goalBlocks:
                    if g['location'] == tupl:
                        self._goalBlocks.remove(g)
        if string_list[0] == "status" and string_list[1] == "of":
            if self._trustBeliefs[member]['rating'] > Trust_Level:
                member = string_list[2][:-1]
                if member != myself:
                    location = message.split('(')[1]
                    location = location.split(')')[0]
                    x = location.split(',')[0]
                    y = location.split(',')[0]
                    location = (int(x), int(y))
                    blocks = []
                    if len(string_list) > 9:
                        block = message.split('{')[1]
                        block = '{' + block.split('}')[0] + '}'
                        block = block.replace("'", '"')
                        block = block.replace("True", "true")
                        block = block.replace("False", "false")
                        block = json.loads(block)
                        blocks.append(block)
                    self._teamObservedStatus[member] = {'location': location, 'is_carrying': blocks,
                                                        'age': self._age - 1}
        if string_list[0] == 'Trustbeliefs:':
            obj = message.split(' ', 1)[1]
            obj = obj.replace("'", '"')
            trust_beliefs = json.loads(obj)
            if self._trustBeliefs[member]['rating'] > Trust_Level:
                for item in trust_beliefs:
                    if item == myself:
                        continue
                    self._trustBeliefs[item]['rating'] = (self._trustBeliefs[item]['rating'] + trust_beliefs[item][
                        'rating']
                                                          * self._trustBeliefs[member]['rating']) / (
                                                                     1 + self._trustBeliefs[member]['rating'])


class StrongAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.PLAN_PATH_TO_ROOM
        self._teamMembers = []

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)
        self._goalBlocks = None
        self._goalsInitialized = False
        self._carrying = None
        self._carrying2 = None
        self._carryingO = None
        self._carryingO2 = None
        self._goalsWrong = []
        self._checkGoals = []
        self._possibleGoalBLocks = []
        self._notExplored = []
        self._trustBeliefs = {}
        self._teamStatus = {}
        self._teamObservedStatus = {}
        self._age = 0

    def filter_observations(self, state):
        self._age += 1
        agent_name = state[self.agent_id]['obj_id']
        headers = ['agent', 'rating', 'age']

        # with open("agents1/agent1.csv", 'w', newline="") as csv_file:
        #     writer = csv.DictWriter(csv_file, fieldnames=headers)
        #     reader = csv.DictReader(csv_file, fieldnames=headers)
        #     writer.writeheader()
        #     csv_file.close()

        df = pd.read_csv("agents1/agent1.csv")
        loc = 0
        if len(self._teamMembers) == 0:
            for member in state['World']['team_members']:
                if member != agent_name and member not in self._teamMembers:
                    self._teamMembers.append(member)
                    self._trustBeliefs[member] = {'rating': df.loc[loc, 'rating'], 'age': self._age}
                    loc += 1


        if self._age % 25 == 0:
            self._sendMessage('Trustbeliefs: ' + str(self._trustBeliefs), agent_name)

        closest_agents = state.get_closest_agents()
        if closest_agents is not None:
            for item in closest_agents:
                name = item['name']
                location = item['location']
                is_carrying = []
                if len(item['is_carrying']) > 0:
                    for block in item['is_carrying']:
                        block = {"size": block['visualization']['size'], "shape": block['visualization']["shape"],
                                 "colour": block['visualization']["colour"]}
                        is_carrying.append(block)
                self._teamObservedStatus[name] = {'location': location, 'is_carrying': is_carrying,
                                                  'age': self._age}
                self._sendMessage('status of ' + name + ': location: '
                                  + str(location) + ', is carrying: ' + str(is_carrying), agent_name)

        receivedMessages = self._processMessages(self._teamMembers)

        for member in self._teamMembers:
            if member in self._teamObservedStatus and self._teamObservedStatus[member] is not None:
                if self._age - self._teamObservedStatus[member]['age'] > delete_age:
                    self._teamObservedStatus[member] = None

        for member in self._teamMembers:
            for message in receivedMessages[member]:
                self._parseMessage(message, member, agent_name)

        # Update trust beliefs for team members
        self._trustBlief(agent_name, state)
        loc1 = 0
        for member in self._teamMembers:
            df.loc[loc1, 'age'] = self._age
            df.loc[loc1, 'rating'] = self._trustBeliefs[member]['rating']
            loc += 1
        df.to_csv("agents1/agent1.csv", index = False)
        return state

    def decide_on_bw4t_action(self, state: State):
        if not self._goalsInitialized:
            self._goalBlocks = state.get_with_property({'is_goal_block': True})
            self._goalsInitialized = True
            self._notExplored = [door for door in state.values()
                                 if 'class_inheritance' in door and 'Door' in door['class_inheritance']]

        agent_name = state[self.agent_id]['obj_id']
        # Add team members
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
                # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)

        # Check if there are objects lying around the agent.
        objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
        if objects is not None:
            for o in objects:
                count = 0
                for g in self._goalBlocks:
                    if o['location'] == g['location']:
                        count += 1
                if count == 0:
                    for g in self._goalBlocks:
                        if len(self.state.get_self()['is_carrying']) == 1 and o['visualization']['shape'] == \
                                self._carryingO['visualization']['shape'] and o['visualization'][
                            'colour'] == \
                                self._carryingO['visualization']['colour']:
                            continue
                        if o['visualization']['shape'] == g['visualization']['shape'] and o['visualization'][
                            'colour'] == \
                                g['visualization']['colour'] and len(o['carried_by']) == 0:
                            if o['location'] not in self._possibleGoalBLocks:
                                self._possibleGoalBLocks.append(o['location'])
                            self._sendMessage('Found goal block {\"size\": ' + str(
                                o['visualization']['size']) + ', \"shape\": ' + str(
                                o['visualization']['shape']) + ', \"colour\": \"' + str(
                                o['visualization']['colour']) + '\"} at location ' + str(o['location']), agent_name)

        while True:
            if Phase.PLAN_PATH_TO_ROOM == self._phase:
                self._navigator.reset_full()
                rooms = [door for door in state.values()
                         if 'class_inheritance' in door and 'Door' in door['class_inheritance']]
                if len(rooms) == 0:
                    return None, {}
                # If not every room has been explored, pick one randomly from the non explored, else pick randomly pick a door
                if len(self._notExplored) != 0:
                    self._door = random.choice(self._notExplored)
                else:
                    self._door = random.choice(rooms)
                doorLoc = self._door['location']
                # Location in front of door is south from door
                doorLoc = doorLoc[0], doorLoc[1] + 1
                # Send message of current action
                self._sendMessage('Moving to ' + self._door['room_name'], agent_name)
                self._navigator.add_waypoints([doorLoc])
                self._phase = Phase.FOLLOW_PATH_TO_ROOM

            if Phase.FOLLOW_PATH_TO_ROOM == self._phase:
                self._state_tracker.update(state)
                # Follow path to room
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                # If the door is not open, open the door.
                if self._door['is_open']:
                    self._phase = Phase.ENTERING_ROOM
                else:
                    self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                self._phase = Phase.WAIT_FOR_DOOR
                # Open door
                self._sendMessage('Opening door of ' + self._door['room_name'], agent_name)
                return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}

            if Phase.WAIT_FOR_DOOR == self._phase:
                self._phase = Phase.ENTERING_ROOM
                return None, {}

            if Phase.ENTERING_ROOM == self._phase:
                if len(self._notExplored) != 0:
                    self._notExplored.remove(self._door)
                # Send message of action
                self._sendMessage('Searching through ' + self._door['room_name'], agent_name)
                self._navigator.reset_full()
                objects = state.get_room_objects(self._door['room_name'])
                # Add all traversable tiles as waypoints
                for o in objects:
                    if o['is_traversable']:
                        self._navigator.add_waypoints([o['location']])
                self._phase = Phase.SEARCHING_ROOM

            if Phase.SEARCHING_ROOM == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                # If there is a goal block in the room pick it up and take it to the goal
                if objects is not None:
                    for o in objects:
                        for g in self._goalBlocks:
                            if o['location'] == g['location']:
                                continue
                            if self._carrying is not None and g == self._carrying:
                                continue
                            if o['visualization']['shape'] == g['visualization']['shape'] and o['visualization'][
                                'colour'] == g['visualization']['colour'] and len(o['carried_by']) == 0:
                                if o['location'] in self._possibleGoalBLocks:
                                    self._possibleGoalBLocks.remove(o['location'])
                                self._sendMessage('Found goal block {\"size\": ' + str(
                                    o['visualization']['size']) + ', \"shape\": ' + str(
                                    o['visualization']['shape']) + ', \"colour\": \"' + str(
                                    o['visualization']['colour']) + '\"} at location ' + str(o['location']), agent_name)
                                self._sendMessage('Picking up goal block {\"size\": ' + str(
                                    o['visualization']['size']) + ', \"shape\": ' + str(
                                    o['visualization']['shape']) + ', \"colour\": \"' + str(
                                    o['visualization']['colour']) + '\"} at location ' + str(
                                    o['location']), agent_name)
                                if len(self.state.get_self()['is_carrying']) == 1:
                                    self._phase = Phase.FOLLOW_PATH_TO_DROP
                                    self._navigator.reset_full()
                                    self._navigator.add_waypoints([g['location']])
                                    self._carrying2 = g
                                    self._carryingO2 = o
                                elif len(self._goalBlocks) == 1 or len(self._notExplored) == 0:
                                    self._phase = Phase.FOLLOW_PATH_TO_DROP
                                    self._navigator.reset_full()
                                    self._navigator.add_waypoints([g['location']])
                                    self._carrying = g
                                    self._carryingO = o
                                else:
                                    self._carrying = g
                                    self._carryingO = o
                                    if len(self._possibleGoalBLocks) == 0:
                                        self._phase = Phase.PLAN_PATH_TO_ROOM
                                    else:
                                        block = self._possibleGoalBLocks[0]
                                        self._navigator.reset_full()
                                        self._navigator.add_waypoints([block])
                                        self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                                action = GrabObject.__name__
                                action_kwargs = {}
                                action_kwargs['object_id'] = o['obj_id']
                                return action, action_kwargs
                # If you explored the entire room, find a new one or go to a known block
                if action != None:
                    return action, {}
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK

            if Phase.FOLLOW_PATH_TO_DROP == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.DROP_OBJECT

            if Phase.DROP_OBJECT == self._phase:
                # If there already is a block in this location, move to a different location and drop the block there.
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        if o['location'] == self.state.get_self()['location']:
                            self._navigator.reset_full()
                            self._navigator.add_waypoints([self._carryingO['location']])
                            self._phase = Phase.FOLLOW_PATH_TO_DROP
                            return None, {}

                self._sendMessage('Dropped goal block {\"size\": ' + str(
                    self._carryingO['visualization']['size']) + ', \"shape\": ' + str(
                    self._carryingO['visualization']['shape']) + ', \"colour\": \"' + str(
                    self._carryingO['visualization']['colour']) + '\"} at location ' + str(
                    self.state.get_self()['location']),
                                  agent_name)

                if len(self.state.get_self()['is_carrying']) == 2:
                    self._goalBlocks.remove(self._carrying2)
                    self._phase = Phase.FOLLOW_PATH_TO_DROP
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([self._carrying['location']])
                    self._carrying2 = None
                    self._carryingO2 = None
                if len(self.state.get_self()['is_carrying']) == 1:
                    self._goalBlocks.remove(self._carrying)
                    self._carrying = None
                    self._carryingO = None

                    if len(self._possibleGoalBLocks) == 0:
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    else:
                        block = self._possibleGoalBLocks[0]
                        self._navigator.reset_full()
                        self._navigator.add_waypoints([block])
                        self._phase = Phase.MOVING_TO_KNOWN_BLOCK

                    # If there are more goal blocks to find, update your goalblock list. Else check if it is a solution
                    if len(self._goalBlocks) >= 1:
                        self._phase = Phase.UPDATE_GOAL_LIST
                        self._navigator.reset_full()
                        self._navigator.add_waypoints([self._goalBlocks[0]['location']])
                        self._checkGoals = []
                        for g in self._goalBlocks:
                            self._checkGoals.append(g)
                    elif len(self._goalBlocks) == 0:
                        self._goalBlocks = state.get_with_property({'is_goal_block': True})
                        self._phase = Phase.CHECK_GOALS

                return DropObject.__name__, {}

            if Phase.CHECK_GOALS == self._phase:
                self._phase = Phase.FOLLOW_PATH_TO_GOAL
                # If there was a wrong block on a goal, try and find a goal block
                if len(self._goalsWrong) != 0:
                    if len(self._possibleGoalBLocks) == 0:
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    else:
                        block = self._possibleGoalBLocks[0]
                        self._navigator.reset_full()
                        self._navigator.add_waypoints([block])
                        self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                    return None, {}
                # If all goal blocks have been checked, try and find a goal block
                if len(self._goalBlocks) == 0:
                    if len(self._possibleGoalBLocks) == 0:
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    else:
                        block = self._possibleGoalBLocks[0]
                        self._navigator.reset_full()
                        self._navigator.add_waypoints([block])
                        self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                    return None, {}
                goal = self._goalBlocks[0]
                self._navigator.reset_full()
                self._navigator.add_waypoints([goal['location']])

            if Phase.FOLLOW_PATH_TO_GOAL == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.PICK_UP_GOAL_BLOCK

            if Phase.PICK_UP_GOAL_BLOCK == self._phase:
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        if o['location'] == self.state.get_self()['location']:
                            # If the current goal location has the wrong block on it, remove it
                            if o['visualization']['shape'] != self._goalBlocks[0]['visualization']['shape'] or \
                                    o['visualization']['colour'] != self._goalBlocks[0]['visualization']['colour']:
                                self._phase = Phase.PUT_AWAY_WRONG_BLOCK
                                self._navigator.reset_full()
                                self._navigator.add_waypoints(
                                    [[self._goalBlocks[0]['location'][0], self._goalBlocks[0]['location'][1] - 3]])
                            # Pick up the goal block that is correct and then immediately drop it
                            else:
                                self._carryingO = o
                                self._phase = Phase.MOVE_GOAL_BLOCK
                            self._sendMessage('Picking up goal block {\"size\": ' + str(
                                o['visualization']['size']) + ', \"shape\": ' + str(
                                o['visualization']['shape']) + ', \"colour\": \"' + str(
                                o['visualization']['colour']) + '\"} at location ' + str(
                                o['location']), agent_name)
                            if o['location'] in self._possibleGoalBLocks:
                                self._possibleGoalBLocks.remove(o['location'])
                            action = GrabObject.__name__
                            self._carryingO = o
                            action_kwargs = {}
                            action_kwargs['object_id'] = o['obj_id']
                            return action, action_kwargs
                # Find a room or block
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK

            if Phase.PUT_AWAY_WRONG_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._goalsWrong.append(self._goalBlocks[0])

                self._phase = Phase.CHECK_GOALS

                return DropObject.__name__, {}

            if Phase.MOVE_GOAL_BLOCK == self._phase:
                self._phase = Phase.CHECK_GOALS
                self._goalBlocks.remove(self._goalBlocks[0])

                self._sendMessage('Dropped goal block {\"size\": ' + str(
                    self._carryingO['visualization']['size']) + ', \"shape\": ' + str(
                    self._carryingO['visualization']['shape']) + ', \"colour\": \"' + str(
                    self._carryingO['visualization']['colour']) + '\"} at location ' + str(
                    self.state.get_self()['location']),
                                  agent_name)
                self._carryingO = None
                return DropObject.__name__, {}

            if Phase.UPDATE_GOAL_LIST == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                if len(self._goalBlocks) == 0:
                    self._goalBlocks = state.get_with_property({'is_goal_block': True})
                    self._phase = Phase.CHECK_GOALS
                    return None, {}
                # If there is a block on the goal, update the goallist
                if len(self._checkGoals) == 0:
                    if len(self._possibleGoalBLocks) == 0:
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    else:
                        block = self._possibleGoalBLocks[0]
                        self._navigator.reset_full()
                        self._navigator.add_waypoints([block])
                        self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                    return None, {}
                goal = self._checkGoals[0]
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        if o['location'] == goal['location']:
                            self._goalBlocks.remove(goal)
                            if len(self._goalBlocks) == 0:
                                self._goalBlocks = state.get_with_property({'is_goal_block': True})
                                self._phase = Phase.CHECK_GOALS
                                return None, {}
                            elif len(self._checkGoals) == 0:
                                if len(self._possibleGoalBLocks) == 0:
                                    self._phase = Phase.PLAN_PATH_TO_ROOM
                                else:
                                    block = self._possibleGoalBLocks[0]
                                    self._navigator.reset_full()
                                    self._navigator.add_waypoints([block])
                                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                            else:
                                next = self._checkGoals[0]
                                self._navigator.reset_full()
                                self._navigator.add_waypoints([next['location']])
                self._checkGoals.remove(goal)

            if Phase.MOVING_TO_KNOWN_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        # If the known block is a goal block, pick it up and bring it to the goal
                        if o['location'] == self._possibleGoalBLocks[0]:
                            for g in self._goalBlocks:
                                if o['visualization']['shape'] == g['visualization']['shape'] and o['visualization'][
                                    'colour'] == g['visualization']['colour']:
                                    self._sendMessage('Found goal block {\"size\": ' + str(
                                        o['visualization']['size']) + ', \"shape\": ' + str(
                                        o['visualization']['shape']) + ', \"colour\": \"' + str(
                                        o['visualization']['colour']) + '\"} at location ' + str(o['location']),
                                                      agent_name)
                                    self._sendMessage('Picking up goal block {\"size\": ' + str(
                                        o['visualization']['size']) + ', \"shape\": ' + str(
                                        o['visualization']['shape']) + ', \"colour\": \"' + str(
                                        o['visualization']['colour']) + '\"} at location ' + str(
                                        o['location']), agent_name)
                                    if len(self.state.get_self()['is_carrying']) == 0:
                                        self._carrying = g
                                        self._carryingO = o
                                    elif len(self.state.get_self()['is_carrying']) == 1:
                                        self._carrying2 = g
                                        self._carryingO2 = o
                                    self._phase = Phase.FOLLOW_PATH_TO_DROP
                                    self._navigator.reset_full()
                                    self._navigator.add_waypoints([g['location']])
                                    action = GrabObject.__name__
                                    action_kwargs = {}
                                    action_kwargs['object_id'] = o['obj_id']
                                    return action, action_kwargs
                self._possibleGoalBLocks.remove(self._possibleGoalBLocks[0])
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK

    def _sendMessage(self, mssg, sender):
        '''
        Enable sending messages in one line of code
        '''
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages:
            self.send_message(msg)

    def _processMessages(self, teamMembers):
        '''
        Process incoming messages and create a dictionary with received messages from each team member.
        '''
        receivedMessages = {}
        for member in teamMembers:
            receivedMessages[member] = []
        for mssg in self.received_messages:
            for member in teamMembers:
                if mssg.from_id == member:
                    receivedMessages[member].append(mssg.content)
        return receivedMessages

    def _trustBlief(self, name, state):
        '''
        Baseline implementation of a trust belief. Creates a dictionary with trust belief scores for each team member, for example based on the received messages.
        '''

        # You can change the default value to your preference

        theta = 0.17
        mu = 0.5
        increment = 0.01

        for member in self._teamMembers:
            if member not in self._teamStatus or member not in self._teamObservedStatus:
                continue
            self._trustBeliefs[member]['age'] = self._age
            rating = self._trustBeliefs[member]['rating']
            if self._teamObservedStatus[member] is not None and self._teamStatus[member]['action'] == 'searching':
                if self._teamObservedStatus[member] is not None:
                    if findRoom(self._teamObservedStatus[member]['location'], state) == self._teamStatus[member][
                        'room']:
                        rating += \
                            increment * (1 / (theta * math.sqrt(2 * math.pi)) *
                                         math.exp(-0.5 * math.pow((rating - mu) / theta, 2)))

            elif self._teamStatus[member]['action'] == 'carrying':
                if self._teamObservedStatus[member] is not None and len(
                        self._teamObservedStatus[member]['is_carrying']) > 0:
                    if self._teamStatus[member]['block'] in self._teamObservedStatus[member]['is_carrying']:
                        rating += \
                            increment * (1 / (theta * math.sqrt(2 * math.pi)) *
                                         math.exp(-0.5 * math.pow((rating - mu) / theta, 2)))

                    else:
                        rating -= \
                            10 * increment * (1 / (theta * math.sqrt(2 * math.pi)) *
                                             math.exp(-0.5 * math.pow((rating - mu) / theta, 2)))
            if rating < 0:
                rating = 0
            if rating > 1:
                rating = 1
            self._trustBeliefs[member]['rating'] = rating

    def _parseMessage(self, message, member, myself):
        string_list = message.split(" ")
        if string_list[0] == "Opening" and string_list[1] == "door":
            room_number = string_list[3].split("_")[1]
            self._teamStatus[member] = {'action': 'opening', 'room': room_number, 'age': self._age}
        if string_list[0] == "Searching" and string_list[1] == "through":
            room_number = string_list[2].split("_")[1]
            self._teamStatus[member] = {'action': 'searching', 'room': room_number, 'age': self._age}
        if string_list[0] == "Found" and string_list[1] == "goal":
            block = message.split('{')[1]
            block = '{' + block.split('}')[0] + '}'
            block = block.replace("'", '"')
            block = block.replace("True", "true")
            block = block.replace("False", "false")
            block = json.loads(block)
            self._teamStatus[member] = {'action': 'finding', 'block': block, 'age': self._age}
            # If the trust is high enough, add goal block to possible goal blocks
            if self._trustBeliefs[member]['rating'] >= Trust_Level and block['colour'] != "" and member != self.agent_name:
                location = message.split('(')[1].split(')')[0].split(', ')
                tupl = (int(location[0]), int(location[1]))
                isgoal = False
                for g in self.state.get_with_property({'is_goal_block': True}):
                    if tupl == g['location']:
                        isgoal = True
                if not isgoal and tupl not in self._possibleGoalBLocks:
                    self._possibleGoalBLocks.append(tupl)
        if string_list[0] == "Picking" and string_list[1] == "up":
            block = message.split('{')[1]
            block = '{' + block.split('}')[0] + '}'
            block = block.replace("'", '"')
            block = block.replace("True", "true")
            block = block.replace("False", "false")
            block = json.loads(block)
            self._teamStatus[member] = {'action': 'carrying', 'block': block, 'age': self._age}
            # If the trust is high enough, add remove goal block from possible goal blocks
            if self._trustBeliefs[member]['rating'] >= Trust_Level and member != self.agent_name:
                location = message.split('(')[1].split(')')[0].split(', ')
                tupl = (int(location[0]), int(location[1]))
                for b in self._possibleGoalBLocks:
                    if b == tupl:
                        self._possibleGoalBLocks.remove(b)
        if string_list[0] == "Dropping" and string_list[1] == "goal":
            block = message.split('{')[1]
            block = '{' + block.split('}')[0] + '}'
            block = block.replace("'", '"')
            block = block.replace("True", "true")
            block = block.replace("False", "false")
            block = json.loads(block)
            self._teamStatus[member] = {'action': 'dropping', 'block': block, 'age': self._age}
            # If the trust is high enough, remove goal block from goalblocks
            if self._trustBeliefs[member]['rating'] >= Trust_Level and member != self.agent_name:
                location = message.split('(')[1].split(')')[0].split(', ')
                tupl = (int(location[0]), int(location[1]))
                for g in self._goalBlocks:
                    if g['location'] == tupl:
                        self._goalBlocks.remove(g)
        if string_list[0] == "status" and string_list[1] == "of":
            if self._trustBeliefs[member]['rating'] > Trust_Level:
                member = string_list[2][:-1]
                if member != myself:
                    location = message.split('(')[1]
                    location = location.split(')')[0]
                    x = location.split(',')[0]
                    y = location.split(',')[0]
                    location = (int(x), int(y))
                    blocks = []
                    if len(string_list) > 9:
                        block = message.split('{')[1]
                        block = '{' + block.split('}')[0] + '}'
                        block = block.replace("'", '"')
                        block = block.replace("True", "true")
                        block = block.replace("False", "false")
                        block = json.loads(block)
                        blocks.append(block)
                    self._teamObservedStatus[member] = {'location': location, 'is_carrying': blocks,
                                                        'age': self._age - 1}
            if string_list[0] == 'Trustbeliefs:':
                obj = message.split(' ', 1)[1]
                obj = obj.replace("'", '"')
                trust_beliefs = json.loads(obj)
                if self._trustBeliefs[member]['rating'] > Trust_Level:
                    for item in trust_beliefs:
                        if item == myself:
                            continue
                        self._trustBeliefs[item]['rating'] = (self._trustBeliefs[item]['rating'] + trust_beliefs[item][
                            'rating']
                                                              * self._trustBeliefs[member]['rating']) / (
                                                                     1 + self._trustBeliefs[member]['rating'])


class ColorblindAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.PLAN_PATH_TO_ROOM
        self._teamMembers = []

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)
        self._goalBlocks = None
        self._goalsInitialized = False
        self._carrying = None
        self._carryingO = None
        self._goalsWrong = []
        self._checkGoals = []
        self._possibleGoalBLocks = []
        self._notExplored = []
        self._trustBeliefs = {}
        self._teamStatus = {}
        self._teamObservedStatus = {}
        self._age = 0

    def filter_observations(self, state):
        self._age += 1
        agent_name = state[self.agent_id]['obj_id']
        loc = 0
        df = pd.read_csv("agents1/agent2.csv")
        if len(self._teamMembers) == 0:
            for member in state['World']['team_members']:
                if member != agent_name and member not in self._teamMembers:
                    self._teamMembers.append(member)
                    self._trustBeliefs[member] = {'rating': df.loc[loc, 'rating'], 'age': self._age}
                    loc += 1

        if self._age % 25 == 0:
            self._sendMessage('Trustbeliefs: ' + str(self._trustBeliefs), agent_name)

        closest_agents = state.get_closest_agents()
        if closest_agents is not None:
            for item in closest_agents:
                name = item['name']
                location = item['location']
                is_carrying = []
                if len(item['is_carrying']) > 0:
                    for block in item['is_carrying']:
                        block = {"size": block['visualization']['size'], "shape": block['visualization']["shape"],
                                 "colour": block['visualization']["colour"]}
                        is_carrying.append(block)
                self._teamObservedStatus[name] = {'location': location, 'is_carrying': is_carrying,
                                                  'age': self._age}
                self._sendMessage('status of ' + name + ': location: '
                                  + str(location) + ', is carrying: ' + str(is_carrying), agent_name)

        receivedMessages = self._processMessages(self._teamMembers)

        for member in self._teamMembers:
            if member in self._teamObservedStatus and self._teamObservedStatus[member] is not None:
                if self._age - self._teamObservedStatus[member]['age'] > delete_age:
                    self._teamObservedStatus[member] = None

        for member in self._teamMembers:
            for message in receivedMessages[member]:
                self._parseMessage(message, member, agent_name)

        # Update trust beliefs for team members
        self._trustBlief(agent_name, state)

        df = pd.read_csv("agents1/agent2.csv")
        loc1 = 0
        for member in self._teamMembers:
            df.loc[loc1, 'age'] = self._age
            df.loc[loc1, 'rating'] = self._trustBeliefs[member]['rating']
            loc += 1
        df.to_csv("agents1/agent2.csv", index=False)
        return state

    def decide_on_bw4t_action(self, state: State):
        if not self._goalsInitialized:
            self._goalBlocks = state.get_with_property({'is_goal_block': True})
            self._goalsInitialized = True
            self._notExplored = [door for door in state.values()
                                 if 'class_inheritance' in door and 'Door' in door['class_inheritance']]

        agent_name = state[self.agent_id]['obj_id']
        # Add team members
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
                # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)

        # Check if there are objects laying around the agent.
        objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
        if objects is not None:
            for o in objects:
                count = 0
                for g in self._goalBlocks:
                    if o['location'] == g['location']:
                        count += 1
                if count == 0:
                    for g in self._goalBlocks:
                        if o['visualization']['shape'] == g['visualization']['shape'] and len(o['carried_by']) == 0:
                            self._sendMessage('Found goal block {\"size\": ' + str(
                                o['visualization']['size']) + ', \"shape\": ' + str(
                                o['visualization']['shape']) + ', \"colour\": \"'
                                              + '\"} at location ' + str(o['location']), agent_name)

        while True:
            if Phase.PLAN_PATH_TO_ROOM == self._phase:
                self._navigator.reset_full()
                rooms = [door for door in state.values()
                         if 'class_inheritance' in door and 'Door' in door['class_inheritance']]
                if len(rooms) == 0:
                    return None, {}
                # If not every room has been explored, pick one randomly from the non explored, else pick randomly pick a door
                if len(self._notExplored) != 0:
                    self._door = random.choice(self._notExplored)
                else:
                    self._door = random.choice(rooms)
                doorLoc = self._door['location']
                # Location in front of door is south from door
                doorLoc = doorLoc[0], doorLoc[1] + 1
                # Send message of current action
                self._sendMessage('Moving to ' + self._door['room_name'], agent_name)
                self._navigator.add_waypoints([doorLoc])
                self._phase = Phase.FOLLOW_PATH_TO_ROOM

            if Phase.FOLLOW_PATH_TO_ROOM == self._phase:
                self._state_tracker.update(state)
                # Follow path to room
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                # If the door is not open, open the door.
                if self._door['is_open']:
                    self._phase = Phase.ENTERING_ROOM
                else:
                    self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                self._phase = Phase.WAIT_FOR_DOOR
                # Open door
                self._sendMessage('Opening door of ' + self._door['room_name'], agent_name)
                return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}

            if Phase.WAIT_FOR_DOOR == self._phase:
                self._phase = Phase.ENTERING_ROOM
                return None, {}

            if Phase.ENTERING_ROOM == self._phase:
                if self._door in self._notExplored:
                    self._notExplored.remove(self._door)
                # Send message of action
                self._sendMessage('Searching through ' + self._door['room_name'], agent_name)
                self._navigator.reset_full()
                objects = state.get_room_objects(self._door['room_name'])
                # Add all traversable tiles as waypoints
                for o in objects:
                    if o['is_traversable']:
                        self._navigator.add_waypoints([o['location']])
                self._phase = Phase.SEARCHING_ROOM

            if Phase.SEARCHING_ROOM == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                # If there is a goal block in the room send a message. Do not pick it up, because you can not see colour
                if objects is not None:
                    for o in objects:
                        for g in self._goalBlocks:
                            if o['visualization']['shape'] == g['visualization']['shape'] and len(o['carried_by']) == 0:
                                self._sendMessage('Found goal block {\"size\": ' + str(
                                    o['visualization']['size']) + ', \"shape\": ' + str(
                                    o['visualization']['shape']) + ', \"colour\": \"' + '\"} at location ' + str(
                                    o['location']), agent_name)
                if action != None:
                    return action, {}

                # If you explored the entire room, find a new one or go to a known block
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK

            if Phase.FOLLOW_PATH_TO_DROP == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.DROP_OBJECT

            if Phase.DROP_OBJECT == self._phase:
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK

                # If there already is a block in this location, move to a different location and drop the block there.
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        if o['location'] == self.state.get_self()['location']:
                            self._navigator.reset_full()
                            self._navigator.add_waypoints([self._carryingO['location']])
                            self._phase = Phase.FOLLOW_PATH_TO_DROP
                            return None, {}

                self._goalBlocks.remove(self._carrying)

                # If there are more goal blocks to find, update your goalblock list. Else check if it is a solution
                if len(self._goalBlocks) >= 1:
                    self._phase = Phase.UPDATE_GOAL_LIST
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([self._goalBlocks[0]['location']])
                    self._checkGoals = []
                    for g in self._goalBlocks:
                        self._checkGoals.append(g)
                elif len(self._goalBlocks) == 0:
                    self._goalBlocks = state.get_with_property({'is_goal_block': True})
                    self._phase = Phase.CHECK_GOALS

                self._sendMessage('Dropped goal block {\"size\": ' + str(
                    self._carryingO['visualization']['size']) + ', \"shape\": ' + str(
                    self._carryingO['visualization']['shape']) + ', \"colour\": \"' + '\"} at location ' + str(
                    self.state.get_self()['location']),
                                  agent_name)

                self._carrying = None
                self._carryingO = None
                return DropObject.__name__, {}

            if Phase.CHECK_GOALS == self._phase:
                self._phase = Phase.FOLLOW_PATH_TO_GOAL
                # If there was a wrong block on a goal, try and find a goal block
                if len(self._goalsWrong) != 0:
                    if len(self._possibleGoalBLocks) == 0:
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    else:
                        block = self._possibleGoalBLocks[0]
                        self._navigator.reset_full()
                        self._navigator.add_waypoints([block])
                        self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                    return None, {}
                # If all goal blocks have been checked, try and find a goal block
                if len(self._goalBlocks) == 0:
                    if len(self._possibleGoalBLocks) == 0:
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    else:
                        block = self._possibleGoalBLocks[0]
                        self._navigator.reset_full()
                        self._navigator.add_waypoints([block])
                        self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                    return None, {}
                goal = self._goalBlocks[0]
                self._navigator.reset_full()
                self._navigator.add_waypoints([goal['location']])

            if Phase.FOLLOW_PATH_TO_GOAL == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.PICK_UP_GOAL_BLOCK

            if Phase.PICK_UP_GOAL_BLOCK == self._phase:
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        if o['location'] == self.state.get_self()['location']:
                            # If the current goal location has the wrong block on it, remove it
                            if o['visualization']['shape'] != self._goalBlocks[0]['visualization']['shape']:
                                self._phase = Phase.PUT_AWAY_WRONG_BLOCK
                                self._navigator.reset_full()
                                self._navigator.add_waypoints(
                                    [[self._goalBlocks[0]['location'][0], self._goalBlocks[0]['location'][1] - 3]])
                            # Pick up the goal block that is correct and then immediately drop it
                            else:
                                self._carryingO = o
                                self._phase = Phase.MOVE_GOAL_BLOCK
                            self._sendMessage('Picking up goal block {\"size\": ' + str(
                                o['visualization']['size']) + ', \"shape\": ' + str(
                                o['visualization']['shape']) + ', \"colour\": \"' + '\"} at location ' + str(
                                o['location']), agent_name)
                            if o['location'] in self._possibleGoalBLocks:
                                self._possibleGoalBLocks.remove(o['location'])
                            action = GrabObject.__name__
                            self._carryingO = o
                            action_kwargs = {}
                            action_kwargs['object_id'] = o['obj_id']
                            return action, action_kwargs
                # Find a room or block
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK

            if Phase.PUT_AWAY_WRONG_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._goalsWrong.append(self._goalBlocks[0])

                self._phase = Phase.CHECK_GOALS

                return DropObject.__name__, {}

            if Phase.MOVE_GOAL_BLOCK == self._phase:
                self._phase = Phase.CHECK_GOALS
                self._goalBlocks.remove(self._goalBlocks[0])

                self._sendMessage('Dropped goal block {\"size\": ' + str(
                    self._carryingO['visualization']['size']) + ', \"shape\": ' + str(
                    self._carryingO['visualization']['shape']) + ', \"colour\": \"' + '\"} at location ' + str(
                    self.state.get_self()['location']),
                                  agent_name)
                self._carryingO = None
                return DropObject.__name__, {}

            if Phase.UPDATE_GOAL_LIST == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                if len(self._goalBlocks) == 0:
                    self._goalBlocks = state.get_with_property({'is_goal_block': True})
                    self._phase = Phase.CHECK_GOALS
                    return None, {}
                # If there is a block on the goal, update the goallist
                if len(self._checkGoals) == 0:
                    if len(self._possibleGoalBLocks) == 0:
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    else:
                        block = self._possibleGoalBLocks[0]
                        self._navigator.reset_full()
                        self._navigator.add_waypoints([block])
                        self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                    return None, {}
                goal = self._checkGoals[0]
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        if o['location'] == goal['location']:
                            self._goalBlocks.remove(goal)
                            if len(self._goalBlocks) == 0:
                                self._goalBlocks = state.get_with_property({'is_goal_block': True})
                                self._phase = Phase.CHECK_GOALS
                                return None, {}
                            elif len(self._checkGoals) == 0:
                                if len(self._possibleGoalBLocks) == 0:
                                    self._phase = Phase.PLAN_PATH_TO_ROOM
                                else:
                                    block = self._possibleGoalBLocks[0]
                                    self._navigator.reset_full()
                                    self._navigator.add_waypoints([block])
                                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                            else:
                                next = self._checkGoals[0]
                                self._navigator.reset_full()
                                self._navigator.add_waypoints([next['location']])
                self._checkGoals.remove(goal)

            if Phase.MOVING_TO_KNOWN_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        # If the known block is a goal block, pick it up and bring it to the goal
                        if o['location'] == self._possibleGoalBLocks[0]:
                            for g in self._goalBlocks:
                                if o['visualization']['shape'] == g['visualization']['shape']:
                                    self._sendMessage('Found goal block {\"size\": ' + str(
                                        o['visualization']['size']) + ', \"shape\": ' + str(
                                        o['visualization']['shape']) + ', \"colour\": \"' + '\"} at location ' + str(
                                        o['location']),
                                                      agent_name)
                                    self._sendMessage('Picking up goal block {\"size\": ' + str(
                                        o['visualization']['size']) + ', \"shape\": ' + str(
                                        o['visualization']['shape']) + ', \"colour\": \"' + '\"} at location ' + str(
                                        o['location']), agent_name)
                                    self._phase = Phase.FOLLOW_PATH_TO_DROP
                                    self._navigator.reset_full()
                                    self._navigator.add_waypoints([g['location']])
                                    action = GrabObject.__name__
                                    action_kwargs = {}
                                    action_kwargs['object_id'] = o['obj_id']
                                    self._carrying = g
                                    self._carryingO = o
                                    return action, action_kwargs
                self._possibleGoalBLocks.remove(self._possibleGoalBLocks[0])
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK

    def _sendMessage(self, mssg, sender):
        '''
        Enable sending messages in one line of code
        '''
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages:
            self.send_message(msg)

    def _processMessages(self, teamMembers):
        '''
        Process incoming messages and create a dictionary with received messages from each team member.
        '''
        receivedMessages = {}
        for member in teamMembers:
            receivedMessages[member] = []
        for mssg in self.received_messages:
            for member in teamMembers:
                if mssg.from_id == member:
                    receivedMessages[member].append(mssg.content)
        return receivedMessages

    def _trustBlief(self, name, state):
        '''
        Baseline implementation of a trust belief. Creates a dictionary with trust belief scores for each team member, for example based on the received messages.
        '''

        # You can change the default value to your preference

        theta = 0.17
        mu = 0.5
        increment = 0.01

        for member in self._teamMembers:
            if member not in self._teamStatus or member not in self._teamObservedStatus:
                continue
            self._trustBeliefs[member]['age'] = self._age
            rating = self._trustBeliefs[member]['rating']
            if self._teamObservedStatus[member] is not None and self._teamStatus[member]['action'] == 'searching':
                if self._teamObservedStatus[member] is not None:
                    if findRoom(self._teamObservedStatus[member]['location'], state) == self._teamStatus[member][
                        'room']:
                        rating += \
                            increment * (1 / (theta * math.sqrt(2 * math.pi)) *
                                         math.exp(-0.5 * math.pow((rating - mu) / theta, 2)))

            elif self._teamStatus[member]['action'] == 'carrying':
                if self._teamObservedStatus[member] is not None and len(
                        self._teamObservedStatus[member]['is_carrying']) > 0:
                    if self._teamStatus[member]['block'] in self._teamObservedStatus[member]['is_carrying']:
                        rating += \
                            increment * (1 / (theta * math.sqrt(2 * math.pi)) *
                                         math.exp(-0.5 * math.pow((rating - mu) / theta, 2)))

                    else:
                        rating -= \
                            10 * increment * (1 / (theta * math.sqrt(2 * math.pi)) *
                                             math.exp(-0.5 * math.pow((rating - mu) / theta, 2)))
            if rating < 0:
                rating = 0
            if rating > 1:
                rating = 1
            self._trustBeliefs[member]['rating'] = rating

    def _parseMessage(self, message, member, myself):
        string_list = message.split(" ")
        if string_list[0] == "Opening" and string_list[1] == "door":
            room_number = string_list[3].split("_")[1]
            self._teamStatus[member] = {'action': 'opening', 'room': room_number, 'age': self._age}
        if string_list[0] == "Searching" and string_list[1] == "through":
            room_number = string_list[2].split("_")[1]
            self._teamStatus[member] = {'action': 'searching', 'room': room_number, 'age': self._age}
        if string_list[0] == "Found" and string_list[1] == "goal":
            block = message.split('{')[1]
            block = '{' + block.split('}')[0] + '}'
            block = block.replace("'", '"')
            block = block.replace("True", "true")
            block = block.replace("False", "false")
            block = json.loads(block)
            self._teamStatus[member] = {'action': 'finding', 'block': block, 'age': self._age}
            # If the trust is high enough, add goal block to possible goal blocks
            if self._trustBeliefs[member]['rating'] >= Trust_Level and block['colour'] != "" and member != self.agent_name:
                location = message.split('(')[1].split(')')[0].split(', ')
                tupl = (int(location[0]), int(location[1]))
                isgoal = False
                for g in self.state.get_with_property({'is_goal_block': True}):
                    if tupl == g['location']:
                        isgoal = True
                if not isgoal and tupl not in self._possibleGoalBLocks:
                    self._possibleGoalBLocks.append(tupl)
        if string_list[0] == "Picking" and string_list[1] == "up":
            block = message.split('{')[1]
            block = '{' + block.split('}')[0] + '}'
            block = block.replace("'", '"')
            block = block.replace("True", "true")
            block = block.replace("False", "false")
            block = json.loads(block)
            self._teamStatus[member] = {'action': 'carrying', 'block': block, 'age': self._age}
            # If the trust is high enough, add remove goal block from possible goal blocks
            if self._trustBeliefs[member]['rating'] >= Trust_Level and member != self.agent_name:
                location = message.split('(')[1].split(')')[0].split(', ')
                tupl = (int(location[0]), int(location[1]))
                for b in self._possibleGoalBLocks:
                    if b == tupl:
                        self._possibleGoalBLocks.remove(b)
        if string_list[0] == "Dropping" and string_list[1] == "goal":
            block = message.split('{')[1]
            block = '{' + block.split('}')[0] + '}'
            block = block.replace("'", '"')
            block = block.replace("True", "true")
            block = block.replace("False", "false")
            block = json.loads(block)
            self._teamStatus[member] = {'action': 'dropping', 'block': block, 'age': self._age}
            # If the trust is high enough, remove goal block from goalblocks
            if self._trustBeliefs[member]['rating'] >= Trust_Level and member != self.agent_name:
                location = message.split('(')[1].split(')')[0].split(', ')
                tupl = (int(location[0]), int(location[1]))
                for g in self._goalBlocks:
                    if g['location'] == tupl:
                        self._goalBlocks.remove(g)
        if string_list[0] == "status" and string_list[1] == "of":
            if self._trustBeliefs[member]['rating'] > Trust_Level:
                member = string_list[2][:-1]
                if member != myself:
                    location = message.split('(')[1]
                    location = location.split(')')[0]
                    x = location.split(',')[0]
                    y = location.split(',')[0]
                    location = (int(x), int(y))
                    blocks = []
                    if len(string_list) > 9:
                        block = message.split('{')[1]
                        block = '{' + block.split('}')[0] + '}'
                        block = block.replace("'", '"')
                        block = block.replace("True", "true")
                        block = block.replace("False", "false")
                        block = json.loads(block)
                        blocks.append(block)
                    self._teamObservedStatus[member] = {'location': location, 'is_carrying': blocks,
                                                        'age': self._age - 1}

        if string_list[0] == 'Trustbeliefs:':
            obj = message.split(' ', 1)[1]
            obj = obj.replace("'", '"')
            trust_beliefs = json.loads(obj)
            if self._trustBeliefs[member]['rating'] > Trust_Level:
                for item in trust_beliefs:
                    if item == myself:
                        continue
                    self._trustBeliefs[item]['rating'] = (self._trustBeliefs[item]['rating'] + trust_beliefs[item][
                        'rating']
                                                          * self._trustBeliefs[member]['rating']) / (
                                                                     1 + self._trustBeliefs[member]['rating'])


class LazyAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.PLAN_PATH_TO_ROOM
        self._teamMembers = []

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)
        self._goalBlocks = None
        self._goalsInitialized = False
        self._carrying = None
        self._carryingO = None
        self._lazymoment = False
        self._prevloc = None
        self._goalsWrong = []
        self._checkGoals = []
        self._possibleGoalBLocks = []
        self._notExplored = []
        self._trustBeliefs = {}
        self._teamStatus = {}
        self._teamObservedStatus = {}
        self._age = 0

    def filter_observations(self, state):
        self._age += 1
        agent_name = state[self.agent_id]['obj_id']

        loc = 0
        df = pd.read_csv("agents1/agent3.csv")
        if len(self._teamMembers) == 0:
            for member in state['World']['team_members']:
                if member != agent_name and member not in self._teamMembers:
                    self._teamMembers.append(member)
                    self._trustBeliefs[member] = {'rating': df.loc[loc, 'rating'], 'age': self._age}
                    loc += 1

        if self._age % 25 == 0:
            self._sendMessage('Trustbeliefs: ' + str(self._trustBeliefs), agent_name)

        closest_agents = state.get_closest_agents()
        if closest_agents is not None:
            for item in closest_agents:
                name = item['name']
                location = item['location']
                is_carrying = []
                if len(item['is_carrying']) > 0:
                    for block in item['is_carrying']:
                        block = {"size": block['visualization']['size'], "shape": block['visualization']["shape"],
                                 "colour": block['visualization']["colour"]}
                        is_carrying.append(block)
                self._teamObservedStatus[name] = {'location': location, 'is_carrying': is_carrying,
                                                  'age': self._age}
                self._sendMessage('status of ' + name + ': location: '
                                  + str(location) + ', is carrying: ' + str(is_carrying), agent_name)

        receivedMessages = self._processMessages(self._teamMembers)

        for member in self._teamMembers:
            if member in self._teamObservedStatus and self._teamObservedStatus[member] is not None:
                if self._age - self._teamObservedStatus[member]['age'] > delete_age:
                    self._teamObservedStatus[member] = None

        for member in self._teamMembers:
            for message in receivedMessages[member]:
                self._parseMessage(message, member, agent_name)

        # Update trust beliefs for team members
        self._trustBlief(agent_name, state)

        df = pd.read_csv("agents1/agent3.csv")
        loc = 0
        for member in self._teamMembers:
            df.loc[loc, 'age'] = self._age
            df.loc[loc, 'rating'] = self._trustBeliefs[member]['rating']
            loc += 1
        df.to_csv("agents1/agent3.csv", index=False)
        return state

    def decide_on_bw4t_action(self, state: State):
        if not self._goalsInitialized:
            self._goalBlocks = state.get_with_property({'is_goal_block': True})
            self._goalsInitialized = True
            self._notExplored = [door for door in state.values()
                                 if 'class_inheritance' in door and 'Door' in door['class_inheritance']]

        agent_name = state[self.agent_id]['obj_id']
        # Add team members
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
                # Process messages from team members
        receivedMessages = self._processMessages(self._teamMembers)

        # Check if there are objects laying around the agent.
        objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
        if objects is not None:
            for o in objects:
                count = 0
                for g in self._goalBlocks:
                    if o['location'] == g['location']:
                        count += 1
                if count == 0:
                    for g in self._goalBlocks:
                        if o['visualization']['shape'] == g['visualization']['shape'] and o['visualization'][
                            'colour'] == \
                                g['visualization']['colour'] and len(o['carried_by']) == 0:
                            if o['location'] not in self._possibleGoalBLocks:
                                self._possibleGoalBLocks.append(o['location'])
                            self._sendMessage('Found goal block {\"size\": ' + str(
                                o['visualization']['size']) + ', \"shape\": ' + str(
                                o['visualization']['shape']) + ', \"colour\": \"' + str(
                                o['visualization']['colour']) + '\"} at location ' + str(o['location']), agent_name)

        while True:
            if Phase.PLAN_PATH_TO_ROOM == self._phase:
                self._navigator.reset_full()
                rooms = [door for door in state.values()
                         if 'class_inheritance' in door and 'Door' in door['class_inheritance']]
                if len(rooms) == 0:
                    return None, {}
                # If not every room has been explored, pick one randomly from the non explored, else pick randomly pick a door
                if len(self._notExplored) != 0:
                    self._door = random.choice(self._notExplored)
                else:
                    self._door = random.choice(rooms)
                doorLoc = self._door['location']
                # Location in front of door is south from door
                doorLoc = doorLoc[0], doorLoc[1] + 1
                # Send message of current action
                self._sendMessage('Moving to ' + self._door['room_name'], agent_name)
                self._navigator.add_waypoints([doorLoc])

                #It doesn't know how many actions it will be to get to the room, so it determines
                #beforehand whether it will abandon the task at some point or not.
                self._lazymoment = False
                if random.random() < 0.5:
                    self._lazymoment = True

                self._phase = Phase.FOLLOW_PATH_TO_ROOM

            if Phase.FOLLOW_PATH_TO_ROOM == self._phase:
                self._state_tracker.update(state)
                # Follow path to room
                # If the agent has previously determined that it will abandon this task
                # every move has a 7% chance of abandoning it at that move
                if self._lazymoment and random.random() < 0.07:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                    return None, {}

                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                # If the agent has previously decided it will abandon this task,
                # But it has not done so yet, then abandon it at the last step before it ends.
                if self._lazymoment:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                    return None, {}
                # If the door is not open, open the door.
                if self._door['is_open']:
                    self._phase = Phase.ENTERING_ROOM
                else:
                    self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                self._phase = Phase.WAIT_FOR_DOOR
                # Open door
                self._sendMessage('Opening door of ' + self._door['room_name'], agent_name)
                # 50% chance of not opening the door and just going to a different room
                if random.random() < 0.5:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                    return None, {}
                return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}

            if Phase.WAIT_FOR_DOOR == self._phase:
                self._phase = Phase.ENTERING_ROOM
                return None, {}

            if Phase.ENTERING_ROOM == self._phase:
                if self._door in self._notExplored:
                    self._notExplored.remove(self._door)
                # Send message of action
                self._sendMessage('Searching through ' + self._door['room_name'], agent_name)
                self._navigator.reset_full()
                # Add all traversable tiles as waypoints
                objects = state.get_room_objects(self._door['room_name'])
                for o in objects:
                    if o['is_traversable']:
                        self._navigator.add_waypoints([o['location']])

                # It doesn't know how many actions it will take to search the room, since it may find an object
                # so it determines beforehand whether it will abandon the task at some point or not.
                self._lazymoment = False
                if random.random() < 0.5:
                    self._lazymoment = True

                self._phase = Phase.SEARCHING_ROOM

            if Phase.SEARCHING_ROOM == self._phase:
                self._state_tracker.update(state)
                # Search the room along the predetermined path
                # If the agent has previously determined that it will abandon this task
                # every move has a 15% chance of abandoning it at that move
                if self._lazymoment and random.random() < 0.15:
                    self._phase = Phase.PLAN_PATH_TO_ROOM

                action = self._navigator.get_move_action(self._state_tracker)
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                # If there is a goal block in the room pick it up and take it to the goal
                if objects is not None:
                    for o in objects:
                        for g in self._goalBlocks:
                            if o['visualization']['shape'] == g['visualization']['shape'] and o['visualization'][
                                'colour'] == g['visualization']['colour'] and len(o['carried_by']) == 0:
                                if o['location'] in self._possibleGoalBLocks:
                                    self._possibleGoalBLocks.remove(o['location'])
                                self._sendMessage('Found goal block {\"size\": ' + str(
                                    o['visualization']['size']) + ', \"shape\": ' + str(
                                    o['visualization']['shape']) + ', \"colour\": \"' + str(
                                    o['visualization']['colour']) + '\"} at location ' + str(o['location']), agent_name)
                                self._sendMessage('Picking up goal block {\"size\": ' + str(
                                    o['visualization']['size']) + ', \"shape\": ' + str(
                                    o['visualization']['shape']) + ', \"colour\": \"' + str(
                                    o['visualization']['colour']) + '\"} at location ' + str(
                                    o['location']), agent_name)
                                if self._lazymoment:
                                    self._phase = Phase.PLAN_PATH_TO_ROOM
                                    return None, {}
                                self._phase = Phase.FOLLOW_PATH_TO_DROP
                                self._navigator.reset_full()
                                self._navigator.add_waypoints([g['location']])
                                action = GrabObject.__name__
                                action_kwargs = {}
                                action_kwargs['object_id'] = o['obj_id']
                                self._carrying = g
                                self._carryingO = o
                                # The agent doesn't know how many actions it will take to carry
                                # the object to the goal, so it determines beforehand whether it
                                # will abandon the task at some point or not.
                                self._lazymoment = False
                                self._prevloc = None
                                if random.random() < 0.5:
                                    self._lazymoment = True
                                return action, action_kwargs
                if action != None:
                    return action, {}

                # If you explored the entire room, find a new one or go to a known block
                if len(self._possibleGoalBLocks) == 0 or self._lazymoment:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK

            if Phase.FOLLOW_PATH_TO_DROP == self._phase:
                self._state_tracker.update(state)
                # Follow the path to the goal
                # If the agent has previously determined that it will abandon this task
                # every move has a 7% chance of abandoning it at that move
                if self._lazymoment and random.random() < 0.07:
                    self._phase = Phase.DROP_OBJECT
                    return None, {}
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    self._prevloc = state.get_self()['location']
                    return action, {}
                # If the agent has previously decided that it will abandon this task,
                # and it has not abandoned it yet, then abandon it now at the last step.
                if self._lazymoment and self._prevloc is not None:
                    # The agent runs back a tile to avoid dropping it in the goal, which would finish its task.
                    self._phase = Phase.RUN_BACK
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([self._prevloc])
                    return None, {}
                self._phase = Phase.DROP_OBJECT

            if Phase.RUN_BACK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.DROP_OBJECT

            if Phase.DROP_OBJECT == self._phase:
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK

                # If there already is a block in this location, move to a different location and drop the block there.
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        if o['location'] == self.state.get_self()['location']:
                            self._navigator.reset_full()
                            self._navigator.add_waypoints([self._carryingO['location']])
                            self._phase = Phase.FOLLOW_PATH_TO_DROP
                            return None, {}

                self._goalBlocks.remove(self._carrying)

                # If there are more goal blocks to find, update your goalblock list. Else check if it is a solution
                if len(self._goalBlocks) >= 1:
                    self._phase = Phase.UPDATE_GOAL_LIST
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([self._goalBlocks[0]['location']])
                    self._checkGoals = []
                    for g in self._goalBlocks:
                        self._checkGoals.append(g)
                elif len(self._goalBlocks) == 0:
                    self._goalBlocks = state.get_with_property({'is_goal_block': True})
                    self._phase = Phase.CHECK_GOALS
                self._sendMessage('Dropped goal block {\"size\": ' + str(
                    self._carryingO['visualization']['size']) + ', \"shape\": ' + str(
                    self._carryingO['visualization']['shape']) + ', \"colour\": \"' + str(
                    self._carryingO['visualization']['colour']) + '\"} at location ' + str(
                    self.state.get_self()['location']),
                                  agent_name)

                self._carrying = None
                self._carryingO = None
                return DropObject.__name__, {}

            if Phase.CHECK_GOALS == self._phase:
                self._phase = Phase.FOLLOW_PATH_TO_GOAL
                # If there was a wrong block on a goal, try and find a goal block
                if len(self._goalsWrong) != 0:
                    if len(self._possibleGoalBLocks) == 0:
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    else:
                        block = self._possibleGoalBLocks[0]
                        self._navigator.reset_full()
                        self._navigator.add_waypoints([block])
                        self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                    return None, {}
                # If all goal blocks have been checked, try and find a goal block
                if len(self._goalBlocks) == 0:
                    if len(self._possibleGoalBLocks) == 0:
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    else:
                        block = self._possibleGoalBLocks[0]
                        self._navigator.reset_full()
                        self._navigator.add_waypoints([block])
                        self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                    return None, {}
                goal = self._goalBlocks[0]
                self._navigator.reset_full()
                self._navigator.add_waypoints([goal['location']])

            if Phase.FOLLOW_PATH_TO_GOAL == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.PICK_UP_GOAL_BLOCK

            if Phase.PICK_UP_GOAL_BLOCK == self._phase:
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        if o['location'] == self.state.get_self()['location']:
                            # If the current goal location has the wrong block on it, remove it
                            if o['visualization']['shape'] != self._goalBlocks[0]['visualization']['shape'] or \
                                    o['visualization']['colour'] != self._goalBlocks[0]['visualization']['colour']:
                                self._phase = Phase.PUT_AWAY_WRONG_BLOCK
                                self._navigator.reset_full()
                                self._navigator.add_waypoints(
                                    [[self._goalBlocks[0]['location'][0], self._goalBlocks[0]['location'][1] - 3]])
                            # Pick up the goal block that is correct and then immediately drop it
                            else:
                                self._carryingO = o
                                self._phase = Phase.MOVE_GOAL_BLOCK
                            self._sendMessage('Picking up goal block {\"size\": ' + str(
                                o['visualization']['size']) + ', \"shape\": ' + str(
                                o['visualization']['shape']) + ', \"colour\": \"' + str(
                                o['visualization']['colour']) + '\"} at location ' + str(
                                o['location']), agent_name)
                            if o['location'] in self._possibleGoalBLocks:
                                self._possibleGoalBLocks.remove(o['location'])
                            action = GrabObject.__name__
                            self._carryingO = o
                            action_kwargs = {}
                            action_kwargs['object_id'] = o['obj_id']
                            return action, action_kwargs
                # Find a room or block
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK

            if Phase.PUT_AWAY_WRONG_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._goalsWrong.append(self._goalBlocks[0])

                self._phase = Phase.CHECK_GOALS

                return DropObject.__name__, {}

            if Phase.MOVE_GOAL_BLOCK == self._phase:
                self._phase = Phase.CHECK_GOALS
                self._goalBlocks.remove(self._goalBlocks[0])

                self._sendMessage('Dropped goal block {\"size\": ' + str(
                    self._carryingO['visualization']['size']) + ', \"shape\": ' + str(
                    self._carryingO['visualization']['shape']) + ', \"colour\": \"' + str(
                    self._carryingO['visualization']['colour']) + '\"} at location ' + str(
                    self.state.get_self()['location']),
                                  agent_name)
                self._carryingO = None
                return DropObject.__name__, {}

            if Phase.UPDATE_GOAL_LIST == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                if len(self._goalBlocks) == 0:
                    self._goalBlocks = state.get_with_property({'is_goal_block': True})
                    self._phase = Phase.CHECK_GOALS
                    return None, {}
                # If there is a block on the goal, update the goallist
                if len(self._checkGoals) == 0:
                    if len(self._possibleGoalBLocks) == 0:
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    else:
                        block = self._possibleGoalBLocks[0]
                        self._navigator.reset_full()
                        self._navigator.add_waypoints([block])
                        self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                    return None, {}
                goal = self._checkGoals[0]
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        if o['location'] == goal['location']:
                            self._goalBlocks.remove(goal)
                            if len(self._goalBlocks) == 0:
                                self._goalBlocks = state.get_with_property({'is_goal_block': True})
                                self._phase = Phase.CHECK_GOALS
                                return None, {}
                            elif len(self._checkGoals) == 0:
                                if len(self._possibleGoalBLocks) == 0:
                                    self._phase = Phase.PLAN_PATH_TO_ROOM
                                else:
                                    block = self._possibleGoalBLocks[0]
                                    self._navigator.reset_full()
                                    self._navigator.add_waypoints([block])
                                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                            else:
                                next = self._checkGoals[0]
                                self._navigator.reset_full()
                                self._navigator.add_waypoints([next['location']])
                self._checkGoals.remove(goal)

            if Phase.MOVING_TO_KNOWN_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        # If the known block is a goal block, pick it up and bring it to the goal
                        if o['location'] == self._possibleGoalBLocks[0]:
                            for g in self._goalBlocks:
                                if o['visualization']['shape'] == g['visualization']['shape'] and o['visualization'][
                                    'colour'] == g['visualization']['colour']:
                                    self._sendMessage('Found goal block {\"size\": ' + str(
                                        o['visualization']['size']) + ', \"shape\": ' + str(
                                        o['visualization']['shape']) + ', \"colour\": \"' + str(
                                        o['visualization']['colour']) + '\"} at location ' + str(o['location']),
                                                      agent_name)
                                    self._sendMessage('Picking up goal block {\"size\": ' + str(
                                        o['visualization']['size']) + ', \"shape\": ' + str(
                                        o['visualization']['shape']) + ', \"colour\": \"' + str(
                                        o['visualization']['colour']) + '\"} at location ' + str(
                                        o['location']), agent_name)
                                    self._phase = Phase.FOLLOW_PATH_TO_DROP
                                    self._navigator.reset_full()
                                    self._navigator.add_waypoints([g['location']])
                                    action = GrabObject.__name__
                                    action_kwargs = {}
                                    action_kwargs['object_id'] = o['obj_id']
                                    self._carrying = g
                                    self._carryingO = o
                                    return action, action_kwargs
                self._possibleGoalBLocks.remove(self._possibleGoalBLocks[0])
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK

    def _sendMessage(self, mssg, sender):
        '''
        Enable sending messages in one line of code
        '''
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages:
            self.send_message(msg)

    def _processMessages(self, teamMembers):
        '''
        Process incoming messages and create a dictionary with received messages from each team member.
        '''
        receivedMessages = {}
        for member in teamMembers:
            receivedMessages[member] = []
        for mssg in self.received_messages:
            for member in teamMembers:
                if mssg.from_id == member:
                    receivedMessages[member].append(mssg.content)
        return receivedMessages

    def _trustBlief(self, name, state):
        '''
        Baseline implementation of a trust belief. Creates a dictionary with trust belief scores for each team member, for example based on the received messages.
        '''

        # You can change the default value to your preference

        theta = 0.17
        mu = 0.5
        increment = 0.01

        for member in self._teamMembers:
            if member not in self._teamStatus or member not in self._teamObservedStatus:
                continue
            self._trustBeliefs[member]['age'] = self._age
            rating = self._trustBeliefs[member]['rating']
            if self._teamObservedStatus[member] is not None and self._teamStatus[member]['action'] == 'searching':
                    if findRoom(self._teamObservedStatus[member]['location'], state) == self._teamStatus[member][
                        'room']:
                        rating += \
                            increment * (1 / (theta * math.sqrt(2 * math.pi)) *
                                         math.exp(-0.5 * math.pow((rating - mu) / theta, 2)))
            elif self._teamStatus[member]['action'] == 'carrying':
                if self._teamObservedStatus[member] is not None and len(
                        self._teamObservedStatus[member]['is_carrying']) > 0:
                    if self._teamStatus[member]['block'] in self._teamObservedStatus[member]['is_carrying']:
                        rating += \
                            increment * (1 / (theta * math.sqrt(2 * math.pi)) *
                                         math.exp(-0.5 * math.pow((rating - mu) / theta, 2)))

                    else:
                        rating -= \
                            10 * increment * (1 / (theta * math.sqrt(2 * math.pi)) *
                                             math.exp(-0.5 * math.pow((rating - mu) / theta, 2)))
            if rating < 0:
                rating = 0
            if rating > 1:
                rating = 1
            self._trustBeliefs[member]['rating'] = rating

    def _parseMessage(self, message, member, myself):
        string_list = message.split(" ")
        if string_list[0] == "Opening" and string_list[1] == "door":
            room_number = string_list[3].split("_")[1]
            self._teamStatus[member] = {'action': 'opening', 'room': room_number, 'age': self._age}
        if string_list[0] == "Searching" and string_list[1] == "through":
            room_number = string_list[2].split("_")[1]
            self._teamStatus[member] = {'action': 'searching', 'room': room_number, 'age': self._age}
        if string_list[0] == "Found" and string_list[1] == "goal":
            block = message.split('{')[1]
            block = '{' + block.split('}')[0] + '}'
            block = block.replace("'", '"')
            block = block.replace("True", "true")
            block = block.replace("False", "false")
            block = json.loads(block)
            self._teamStatus[member] = {'action': 'finding', 'block': block, 'age': self._age}
            # If the trust is high enough, add goal block to possible goal blocks
            if self._trustBeliefs[member]['rating'] >= Trust_Level and block['colour'] != "" and member != self.agent_name:
                location = message.split('(')[1].split(')')[0].split(', ')
                tupl = (int(location[0]), int(location[1]))
                isgoal = False
                for g in self.state.get_with_property({'is_goal_block': True}):
                    if tupl == g['location']:
                        isgoal = True
                if not isgoal and tupl not in self._possibleGoalBLocks:
                    self._possibleGoalBLocks.append(tupl)
        if string_list[0] == "Picking" and string_list[1] == "up":
            block = message.split('{')[1]
            block = '{' + block.split('}')[0] + '}'
            block = block.replace("'", '"')
            block = block.replace("True", "true")
            block = block.replace("False", "false")
            block = json.loads(block)
            self._teamStatus[member] = {'action': 'carrying', 'block': block, 'age': self._age}
            # If the trust is high enough, add remove goal block from possible goal blocks
            if self._trustBeliefs[member]['rating'] >= Trust_Level and member != self.agent_name:
                location = message.split('(')[1].split(')')[0].split(', ')
                tupl = (int(location[0]), int(location[1]))
                for b in self._possibleGoalBLocks:
                    if b == tupl:
                        self._possibleGoalBLocks.remove(b)
        if string_list[0] == "Dropping" and string_list[1] == "goal":
            block = message.split('{')[1]
            block = '{' + block.split('}')[0] + '}'
            block = block.replace("'", '"')
            block = block.replace("True", "true")
            block = block.replace("False", "false")
            block = json.loads(block)
            self._teamStatus[member] = {'action': 'dropping', 'block': block, 'age': self._age}
            # If the trust is high enough, remove goal block from goalblocks
            if self._trustBeliefs[member]['rating'] >= Trust_Level and member != self.agent_name:
                location = message.split('(')[1].split(')')[0].split(', ')
                tupl = (int(location[0]), int(location[1]))
                for g in self._goalBlocks:
                    if g['location'] == tupl:
                        self._goalBlocks.remove(g)
        if string_list[0] == "status" and string_list[1] == "of":
            if self._trustBeliefs[member]['rating'] > Trust_Level:
                member = string_list[2][:-1]
                if member != myself:
                    location = message.split('(')[1]
                    location = location.split(')')[0]
                    x = location.split(',')[0]
                    y = location.split(',')[0]
                    location = (int(x), int(y))
                    blocks = []
                    if len(string_list) > 9:
                        block = message.split('{')[1]
                        block = '{' + block.split('}')[0] + '}'
                        block = block.replace("'", '"')
                        block = block.replace("True", "true")
                        block = block.replace("False", "false")
                        block = json.loads(block)
                        blocks.append(block)
                    self._teamObservedStatus[member] = {'location': location, 'is_carrying': blocks,
                                                        'age': self._age - 1}

        if string_list[0] == 'Trustbeliefs:':
            obj = message.split(' ', 1)[1]
            obj = obj.replace("'", '"')
            trust_beliefs = json.loads(obj)
            if self._trustBeliefs[member]['rating'] > Trust_Level:
                for item in trust_beliefs:
                    if item == myself:
                        continue
                    self._trustBeliefs[item]['rating'] = (self._trustBeliefs[item]['rating'] + trust_beliefs[item][
                        'rating']
                                                          * self._trustBeliefs[member]['rating']) / (
                                                                     1 + self._trustBeliefs[member]['rating'])


class LiarAgent(BW4TBrain):

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._phase = Phase.PLAN_PATH_TO_ROOM
        self._teamMembers = []

    def initialize(self):
        super().initialize()
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id,
                                    action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)
        self._goalBlocks = None
        self._goalsInitialized = False
        self._carrying = None
        self._carryingO = None
        self._goalsWrong = []
        self._checkGoals = []
        self._possibleGoalBLocks = []
        self._notExplored = []
        self._trustBeliefs = {}
        self._teamStatus = {}
        self._teamObservedStatus = {}
        self._age = 0

    def filter_observations(self, state):
        self._age += 1
        agent_name = state[self.agent_id]['obj_id']
        loc = 0
        df = pd.read_csv("agents1/agent4.csv")

        if len(self._teamMembers) == 0:
            for member in state['World']['team_members']:
                if member != agent_name and member not in self._teamMembers:
                    self._teamMembers.append(member)
                    self._trustBeliefs[member] = {'rating': df.loc[loc, 'rating'], 'age': self._age}
                    loc += 1

        if self._age % 25 == 0:
            self._sendMessage('Trustbeliefs: ' + str(self._trustBeliefs), agent_name)

        closest_agents = state.get_closest_agents()
        if closest_agents is not None:
            for item in closest_agents:
                name = item['name']
                location = item['location']
                is_carrying = []
                if len(item['is_carrying']) > 0:
                    for block in item['is_carrying']:
                        block = {"size": block['visualization']['size'], "shape": block['visualization']["shape"],
                                 "colour": block['visualization']["colour"]}
                        is_carrying.append(block)
                self._teamObservedStatus[name] = {'location': location, 'is_carrying': is_carrying,
                                                  'age': self._age}
                if self._goalsInitialized:
                    self._sendMassage('status of ' + name + ': location: '
                                      + str(location) + ', is carrying: ' + str(is_carrying), agent_name)
                else:
                    self._sendMessage('status of ' + name + ': location: '
                                      + str(location) + ', is carrying: ' + str(is_carrying), agent_name)

        receivedMessages = self._processMessages(self._teamMembers)

        for member in self._teamMembers:
            if member in self._teamObservedStatus and self._teamObservedStatus[member] is not None:
                if self._age - self._teamObservedStatus[member]['age'] > delete_age:
                    self._teamObservedStatus[member] = None

        for member in self._teamMembers:
            for message in receivedMessages[member]:
                self._parseMessage(message, member, agent_name)

        # Update trust beliefs for team members
        self._trustBlief(agent_name, state)

        df = pd.read_csv("agents1/agent4.csv")
        loc = 0
        for member in self._teamMembers:
            df.loc[loc, 'age'] = self._age
            df.loc[loc, 'rating'] = self._trustBeliefs[member]['rating']
            loc += 1
        df.to_csv("agents1/agent4.csv", index=False)

        return state

    def decide_on_bw4t_action(self, state: State):
        if not self._goalsInitialized:
            self._goalBlocks = state.get_with_property({'is_goal_block': True})
            self._goalsInitialized = True
            self._notExplored = [door for door in state.values()
                                 if 'class_inheritance' in door and 'Door' in door['class_inheritance']]
            # predetermine the roomnames, goals and agent names once,
            # such that the state doesn't have to be passed to the sendmassage method
            self._all_room_names = state.get_all_room_names()
            if 'world_bounds' in self._all_room_names:
                self._all_room_names.remove('world_bounds')
            self._all_goals = state.get_with_property({'is_goal_block': True})
            self._all_agents = state.get_agents()

        # set the agent's own state object every tick,
        # such that the state doesn't have to be passed to the sendmassage method
        self._memyself = state.get_self()

        agent_name = state[self.agent_id]['obj_id']
        # Add team members
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._teamMembers:
                self._teamMembers.append(member)
                # Process messages from team members

        # Check if there are objects laying around the agent.
        objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
        if objects is not None:
            for o in objects:
                count = 0
                for g in self._goalBlocks:
                    if o['location'] == g['location']:
                        count += 1
                if count == 0:
                    for g in self._goalBlocks:
                        if o['visualization']['shape'] == g['visualization']['shape'] and o['visualization'][
                            'colour'] == \
                                g['visualization']['colour'] and len(o['carried_by']) == 0:
                            if o['location'] not in self._possibleGoalBLocks:
                                self._possibleGoalBLocks.append(o['location'])
                            self._sendMassage('Found goal block {\"size\": ' + str(
                                o['visualization']['size']) + ', \"shape\": ' + str(
                                o['visualization']['shape']) + ', \"colour\": \"' + str(
                                o['visualization']['colour']) + '\"} at location ' + str(o['location']), agent_name)

        while True:
            if Phase.PLAN_PATH_TO_ROOM == self._phase:
                self._navigator.reset_full()
                rooms = [door for door in state.values()
                         if 'class_inheritance' in door and 'Door' in door['class_inheritance']]
                if len(rooms) == 0:
                    return None, {}
                # If not every room has been explored, pick one randomly from the non explored, else pick randomly pick a door
                if len(self._notExplored) != 0:
                    self._door = random.choice(self._notExplored)
                else:
                    self._door = random.choice(rooms)
                doorLoc = self._door['location']
                # Location in front of door is south from door
                doorLoc = doorLoc[0], doorLoc[1] + 1
                # Send message of current action
                self._sendMassage('Moving to ' + self._door['room_name'], agent_name)
                self._navigator.add_waypoints([doorLoc])
                self._phase = Phase.FOLLOW_PATH_TO_ROOM

            if Phase.FOLLOW_PATH_TO_ROOM == self._phase:
                self._state_tracker.update(state)
                # Follow path to room
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                # If the door is not open, open the door.
                if self._door['is_open']:
                    self._phase = Phase.ENTERING_ROOM
                else:
                    self._phase = Phase.OPEN_DOOR

            if Phase.OPEN_DOOR == self._phase:
                self._phase = Phase.WAIT_FOR_DOOR
                # Open door
                self._sendMassage('Opening door of ' + self._door['room_name'], agent_name)
                return OpenDoorAction.__name__, {'object_id': self._door['obj_id']}

            if Phase.WAIT_FOR_DOOR == self._phase:
                self._phase = Phase.ENTERING_ROOM
                return None, {}

            if Phase.ENTERING_ROOM == self._phase:
                if self._door in self._notExplored:
                    self._notExplored.remove(self._door)
                # Send message of action
                self._sendMassage('Searching through ' + self._door['room_name'], agent_name)
                self._navigator.reset_full()
                objects = state.get_room_objects(self._door['room_name'])
                # Add all traversable tiles as waypoints
                for o in objects:
                    if o['is_traversable']:
                        self._navigator.add_waypoints([o['location']])
                self._phase = Phase.SEARCHING_ROOM

            if Phase.SEARCHING_ROOM == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                # If there is a goal block in the room pick it up and take it to the goal
                if objects is not None:
                    for o in objects:
                        for g in self._goalBlocks:
                            if o['visualization']['shape'] == g['visualization']['shape'] and o['visualization'][
                                'colour'] == g['visualization']['colour'] and len(o['carried_by']) == 0:
                                if o['location'] in self._possibleGoalBLocks:
                                    self._possibleGoalBLocks.remove(o['location'])
                                self._sendMassage('Found goal block {\"size\": ' + str(
                                    o['visualization']['size']) + ', \"shape\": ' + str(
                                    o['visualization']['shape']) + ', \"colour\": \"' + str(
                                    o['visualization']['colour']) + '\"} at location ' + str(o['location']), agent_name)
                                self._sendMassage('Picking up goal block {\"size\": ' + str(
                                    o['visualization']['size']) + ', \"shape\": ' + str(
                                    o['visualization']['shape']) + ', \"colour\": \"' + str(
                                    o['visualization']['colour']) + '\"} at location ' + str(
                                    o['location']), agent_name)
                                self._phase = Phase.FOLLOW_PATH_TO_DROP
                                self._navigator.reset_full()
                                self._navigator.add_waypoints([g['location']])
                                action = GrabObject.__name__
                                action_kwargs = {}
                                action_kwargs['object_id'] = o['obj_id']
                                self._carrying = g
                                self._carryingO = o
                                return action, action_kwargs
                if action != None:
                    return action, {}

                # If you explored the entire room, find a new one or go to a known block
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK

            if Phase.FOLLOW_PATH_TO_DROP == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.DROP_OBJECT

            if Phase.DROP_OBJECT == self._phase:
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK

                # If there already is a block in this location, move to a different location and drop the block there.
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        if o['location'] == self.state.get_self()['location']:
                            self._navigator.reset_full()
                            self._navigator.add_waypoints([self._carryingO['location']])
                            self._phase = Phase.FOLLOW_PATH_TO_DROP
                            return None, {}

                self._goalBlocks.remove(self._carrying)
                self._sendMassage('Removing block' + str(self._carrying) + ' from list: ' + str(self._goalBlocks),
                                  agent_name)

                # If there are more goal blocks to find, update your goalblock list. Else check if it is a solution
                if len(self._goalBlocks) >= 1:
                    self._sendMassage('Updating goal list with ' + str(len(self._goalBlocks)), agent_name)
                    self._phase = Phase.UPDATE_GOAL_LIST
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([self._goalBlocks[0]['location']])
                    self._checkGoals = []
                    for g in self._goalBlocks:
                        self._checkGoals.append(g)
                elif len(self._goalBlocks) == 0:
                    self._goalBlocks = state.get_with_property({'is_goal_block': True})
                    self._phase = Phase.CHECK_GOALS

                self._sendMassage('Dropped goal block {\"size\": ' + str(
                    self._carryingO['visualization']['size']) + ', \"shape\": ' + str(
                    self._carryingO['visualization']['shape']) + ', \"colour\": \"' + str(
                    self._carryingO['visualization']['colour']) + '\"} at location ' + str(
                    self.state.get_self()['location']),
                                  agent_name)

                self._carrying = None
                self._carryingO = None
                return DropObject.__name__, {}

            if Phase.CHECK_GOALS == self._phase:
                self._phase = Phase.FOLLOW_PATH_TO_GOAL
                # If there was a wrong block on a goal, try and find a goal block
                if len(self._goalsWrong) != 0:
                    if len(self._possibleGoalBLocks) == 0:
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    else:
                        block = self._possibleGoalBLocks[0]
                        self._navigator.reset_full()
                        self._navigator.add_waypoints([block])
                        self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                    return None, {}
                # If all goal blocks have been checked, try and find a goal block
                if len(self._goalBlocks) == 0:
                    if len(self._possibleGoalBLocks) == 0:
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    else:
                        block = self._possibleGoalBLocks[0]
                        self._navigator.reset_full()
                        self._navigator.add_waypoints([block])
                        self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                    return None, {}
                goal = self._goalBlocks[0]
                self._navigator.reset_full()
                self._navigator.add_waypoints([goal['location']])

            if Phase.FOLLOW_PATH_TO_GOAL == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._phase = Phase.PICK_UP_GOAL_BLOCK

            if Phase.PICK_UP_GOAL_BLOCK == self._phase:
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        if o['location'] == self.state.get_self()['location']:
                            # If the current goal location has the wrong block on it, remove it
                            if o['visualization']['shape'] != self._goalBlocks[0]['visualization']['shape'] or \
                                    o['visualization']['colour'] != self._goalBlocks[0]['visualization']['colour']:
                                self._phase = Phase.PUT_AWAY_WRONG_BLOCK
                                self._navigator.reset_full()
                                self._navigator.add_waypoints(
                                    [[self._goalBlocks[0]['location'][0], self._goalBlocks[0]['location'][1] - 3]])
                            # Pick up the goal block that is correct and then immediately drop it
                            else:
                                self._carryingO = o
                                self._phase = Phase.MOVE_GOAL_BLOCK
                            self._sendMassage('Picking up goal block {\"size\": ' + str(
                                o['visualization']['size']) + ', \"shape\": ' + str(
                                o['visualization']['shape']) + ', \"colour\": \"' + str(
                                o['visualization']['colour']) + '\"} at location ' + str(
                                o['location']), agent_name)
                            if o['location'] in self._possibleGoalBLocks:
                                self._possibleGoalBLocks.remove(o['location'])
                            action = GrabObject.__name__
                            self._carryingO = o
                            action_kwargs = {}
                            action_kwargs['object_id'] = o['obj_id']
                            return action, action_kwargs
                # Find a room or block
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK

            if Phase.PUT_AWAY_WRONG_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                self._goalsWrong.append(self._goalBlocks[0])

                self._phase = Phase.CHECK_GOALS

                return DropObject.__name__, {}

            if Phase.MOVE_GOAL_BLOCK == self._phase:
                self._phase = Phase.CHECK_GOALS
                self._goalBlocks.remove(self._goalBlocks[0])

                self._sendMassage('Dropped goal block {\"size\": ' + str(
                    self._carryingO['visualization']['size']) + ', \"shape\": ' + str(
                    self._carryingO['visualization']['shape']) + ', \"colour\": \"' + str(
                    self._carryingO['visualization']['colour']) + '\"} at location ' + str(
                    self.state.get_self()['location']),
                                  agent_name)
                self._carryingO = None
                return DropObject.__name__, {}

            if Phase.UPDATE_GOAL_LIST == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                for g in self._goalBlocks:
                    self._sendMassage('Goal block {\"size\": ' + str(
                        g['visualization']['size']) + ', \"shape\": ' + str(
                        g['visualization']['shape']) + ', \"colour\": \"' + str(
                        g['visualization']['colour']) + '\"} at location ' + str(
                        self.state.get_self()['location']),
                                      agent_name)
                if len(self._goalBlocks) == 0:
                    self._goalBlocks = state.get_with_property({'is_goal_block': True})
                    self._phase = Phase.CHECK_GOALS
                    return None, {}
                # If there is a block on the goal, update the goallist
                if len(self._checkGoals) == 0:
                    if len(self._possibleGoalBLocks) == 0:
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    else:
                        block = self._possibleGoalBLocks[0]
                        self._navigator.reset_full()
                        self._navigator.add_waypoints([block])
                        self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                    return None, {}
                goal = self._checkGoals[0]
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        if o['location'] == goal['location']:
                            self._goalBlocks.remove(goal)
                            if len(self._goalBlocks) == 0:
                                self._goalBlocks = state.get_with_property({'is_goal_block': True})
                                self._phase = Phase.CHECK_GOALS
                                return None, {}
                            elif len(self._checkGoals) == 0:
                                if len(self._possibleGoalBLocks) == 0:
                                    self._phase = Phase.PLAN_PATH_TO_ROOM
                                else:
                                    block = self._possibleGoalBLocks[0]
                                    self._navigator.reset_full()
                                    self._navigator.add_waypoints([block])
                                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK
                            else:
                                next = self._checkGoals[0]
                                self._navigator.reset_full()
                                self._navigator.add_waypoints([next['location']])
                self._checkGoals.remove(goal)

            if Phase.MOVING_TO_KNOWN_BLOCK == self._phase:
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    return action, {}
                objects = state.get_closest_with_property({'class_inheritance': ['CollectableBlock']})
                if objects is not None:
                    for o in objects:
                        # If the known block is a goal block, pick it up and bring it to the goal
                        if o['location'] == self._possibleGoalBLocks[0]:
                            for g in self._goalBlocks:
                                if o['visualization']['shape'] == g['visualization']['shape'] and o['visualization'][
                                    'colour'] == g['visualization']['colour']:
                                    self._sendMassage('Found goal block {\"size\": ' + str(
                                        o['visualization']['size']) + ', \"shape\": ' + str(
                                        o['visualization']['shape']) + ', \"colour\": \"' + str(
                                        o['visualization']['colour']) + '\"} at location ' + str(o['location']),
                                                      agent_name)
                                    self._sendMassage('Picking up goal block {\"size\": ' + str(
                                        o['visualization']['size']) + ', \"shape\": ' + str(
                                        o['visualization']['shape']) + ', \"colour\": \"' + str(
                                        o['visualization']['colour']) + '\"} at location ' + str(
                                        o['location']), agent_name)
                                    self._phase = Phase.FOLLOW_PATH_TO_DROP
                                    self._navigator.reset_full()
                                    self._navigator.add_waypoints([g['location']])
                                    action = GrabObject.__name__
                                    action_kwargs = {}
                                    action_kwargs['object_id'] = o['obj_id']
                                    self._carrying = g
                                    self._carryingO = o
                                    return action, action_kwargs
                self._possibleGoalBLocks.remove(self._possibleGoalBLocks[0])
                if len(self._possibleGoalBLocks) == 0:
                    self._phase = Phase.PLAN_PATH_TO_ROOM
                else:
                    block = self._possibleGoalBLocks[0]
                    self._navigator.reset_full()
                    self._navigator.add_waypoints([block])
                    self._phase = Phase.MOVING_TO_KNOWN_BLOCK

    def _sendMessage(self, mssg, sender):
        '''
        Enable sending messages in one line of code
        '''
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages:
            self.send_message(msg)

    def _processMessages(self, teamMembers):
        '''
        Process incoming messages and create a dictionary with received messages from each team member.
        '''
        receivedMessages = {}
        for member in teamMembers:
            receivedMessages[member] = []
        for mssg in self.received_messages:
            for member in teamMembers:
                if mssg.from_id == member:
                    receivedMessages[member].append(mssg.content)
        return receivedMessages

    def _trustBlief(self, name, state):
        '''
        Baseline implementation of a trust belief. Creates a dictionary with trust belief scores for each team member, for example based on the received messages.
        '''

        # You can change the default value to your preference

        theta = 0.17
        mu = 0.5
        increment = 0.01

        for member in self._teamMembers:
            if member not in self._teamStatus or member not in self._teamObservedStatus:
                continue
            self._trustBeliefs[member]['age'] = self._age
            rating = self._trustBeliefs[member]['rating']
            if self._teamObservedStatus[member] is not None and self._teamStatus[member]['action'] == 'searching':
                if self._teamObservedStatus[member] is not None:
                    if findRoom(self._teamObservedStatus[member]['location'], state) == self._teamStatus[member][
                            'room']:
                        rating += \
                            increment * (1 / (theta * math.sqrt(2 * math.pi)) *
                                         math.exp(-0.5 * math.pow((rating - mu) / theta, 2)))

            elif self._teamStatus[member]['action'] == 'carrying':
                if self._teamObservedStatus[member] is not None and len(
                        self._teamObservedStatus[member]['is_carrying']) > 0:
                    if self._teamStatus[member]['block'] in self._teamObservedStatus[member]['is_carrying']:
                        rating += \
                            increment * (1 / (theta * math.sqrt(2 * math.pi)) *
                                         math.exp(-0.5 * math.pow((rating - mu) / theta, 2)))
                    else:
                        rating -= \
                            10 * increment * (1 / (theta * math.sqrt(2 * math.pi)) *
                                             math.exp(-0.5 * math.pow((rating - mu) / theta, 2)))
            if rating < 0:
                rating = 0
            if rating > 1:
                rating = 1
            self._trustBeliefs[member]['rating'] = rating

    def _parseMessage(self, message, member, myself):
        string_list = message.split(" ")
        if string_list[0] == "Opening" and string_list[1] == "door":
            room_number = string_list[3].split("_")[1]
            self._teamStatus[member] = {'action': 'opening', 'room': room_number, 'age': self._age}
        if string_list[0] == "Searching" and string_list[1] == "through":
            room_number = string_list[2].split("_")[1]
            self._teamStatus[member] = {'action': 'searching', 'room': room_number, 'age': self._age}
        if string_list[0] == "Found" and string_list[1] == "goal":
            block = message.split('{')[1]
            block = '{' + block.split('}')[0] + '}'
            block = block.replace("'", '"')
            block = block.replace("True", "true")
            block = block.replace("False", "false")
            block = json.loads(block)
            self._teamStatus[member] = {'action': 'finding', 'block': block, 'age': self._age}
            # If the trust is high enough, add goal block to possible goal blocks
            if self._trustBeliefs[member]['rating'] >= Trust_Level and block['colour'] != "" and member != self.agent_name:
                location = message.split('(')[1].split(')')[0].split(', ')
                tupl = (int(location[0]), int(location[1]))
                isgoal = False
                for g in self.state.get_with_property({'is_goal_block': True}):
                    if tupl == g['location']:
                        isgoal = True
                if not isgoal and tupl not in self._possibleGoalBLocks:
                    self._possibleGoalBLocks.append(tupl)
        if string_list[0] == "Picking" and string_list[1] == "up":
            block = message.split('{')[1]
            block = '{' + block.split('}')[0] + '}'
            block = block.replace("'", '"')
            block = block.replace("True", "true")
            block = block.replace("False", "false")
            block = json.loads(block)
            self._teamStatus[member] = {'action': 'carrying', 'block': block, 'age': self._age}
            # If the trust is high enough, add remove goal block from possible goal blocks
            if self._trustBeliefs[member]['rating'] >= Trust_Level and member != self.agent_name:
                location = message.split('(')[1].split(')')[0].split(', ')
                tupl = (int(location[0]), int(location[1]))
                for b in self._possibleGoalBLocks:
                    if b == tupl:
                        self._possibleGoalBLocks.remove(b)
        if string_list[0] == "Dropping" and string_list[1] == "goal":
            block = message.split('{')[1]
            block = '{' + block.split('}')[0] + '}'
            block = block.replace("'", '"')
            block = block.replace("True", "true")
            block = block.replace("False", "false")
            block = json.loads(block)
            self._teamStatus[member] = {'action': 'dropping', 'block': block, 'age': self._age}
            # If the trust is high enough, remove goal block from goalblocks
            if self._trustBeliefs[member]['rating'] >= Trust_Level and member != self.agent_name:
                location = message.split('(')[1].split(')')[0].split(', ')
                tupl = (int(location[0]), int(location[1]))
                for g in self._goalBlocks:
                    if g['location'] == tupl:
                        self._goalBlocks.remove(g)
        if string_list[0] == "status" and string_list[1] == "of":
            if self._trustBeliefs[member]['rating'] > Trust_Level:
                member = string_list[2][:-1]
                if member != myself:
                    location = message.split('(')[1]
                    location = location.split(')')[0]
                    x = location.split(',')[0]
                    y = location.split(',')[0]
                    location = (int(x), int(y))
                    blocks = []
                    if len(string_list) > 9:
                        block = message.split('{')[1]
                        block = '{' + block.split('}')[0] + '}'
                        block = block.replace("'", '"')
                        block = block.replace("True", "true")
                        block = block.replace("False", "false")
                        block = json.loads(block)
                        blocks.append(block)
                    self._teamObservedStatus[member] = {'location': location, 'is_carrying': blocks,
                                                        'age': self._age - 1}

        if string_list[0] == 'Trustbeliefs:':
            obj = message.split(' ', 1)[1]
            obj = obj.replace("'", '"')
            trust_beliefs = json.loads(obj)
            if self._trustBeliefs[member]['rating'] > Trust_Level:
                for item in trust_beliefs:
                    if item == myself:
                        continue
                    self._trustBeliefs[item]['rating'] = (self._trustBeliefs[item]['rating'] + trust_beliefs[item][
                        'rating']
                                                          * self._trustBeliefs[member]['rating']) / (
                                                                     1 + self._trustBeliefs[member]['rating'])

    # Returns a random lie
    def _generateLie(self):
        # Select one of the 7 random lies
        rand = random.randint(1, 7)
        # lie about moving to a room
        if rand == 1:
            return 'Moving to ' + random.choice(self._all_room_names)
        # lie about searching through a room
        if rand == 2:
            return 'Searching through ' + random.choice(self._all_room_names)
        # lie about opening a door of a room
        if rand == 3:
            return 'Opening door of ' + random.choice(self._all_room_names)
        # lie about having found a goal block
        if rand == 4:
            o = random.choice(self._all_goals)
            memyself = self._memyself
            return 'Found goal block {\"size\": ' + str(
                o['visualization']['size']) + ', \"shape\": ' + str(
                o['visualization']['shape']) + ', \"colour\": \"' + str(
                o['visualization']['colour']) + '\"} at location ' + str(memyself['location'])
        # lie about picking up a goal block
        if rand == 5:
            o = random.choice(self._all_goals)
            memyself = self._memyself
            return 'Picking up goal block {\"size\": ' + str(
                o['visualization']['size']) + ', \"shape\": ' + str(
                o['visualization']['shape']) + ', \"colour\": \"' + str(
                o['visualization']['colour']) + '\"} at location ' + str(
                memyself['location'])
        # lie about dropping a goal block
        if rand == 6:
            o = random.choice(self._all_goals)
            return 'Dropped goal block {\"size\": ' + str(
                o['visualization']['size']) + ', \"shape\": ' + str(
                o['visualization']['shape']) + ', \"colour\": \"' + str(
                o['visualization']['colour']) + '\"} at location ' + str(
                o['location'])
        # lie about observing another agent
        if rand == 7:
            agent = random.choice(self._all_agents)
            loc = self._memyself['location']
            goalvis = random.choice(self._all_goals)['visualization']
            return 'status of ' + agent['name'] + ': location: ' + str(loc) + ', is carrying: ' + str(goalvis)

    # The liar agent always call this function instead of _sendMessage, to incorporate its lies
    def _sendMassage(self, mssg, sender):
        msssg = mssg
        # it has 80% chance of using the lie
        # 20% chance of using the normal message
        if random.random() < 0.8:
            msssg = self._generateLie()
        self._sendMessage(msssg, sender)
