#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import random
from array import array
import statistics


# # Board

# In[2]:


expansionPack = False


# In[3]:


numWoods = 4 + 2 * expansionPack
numBricks = 3 + 2 * expansionPack
numSheeps = 4 + 2 * expansionPack
numWheats = 4 + 2 * expansionPack
numRocks = 3 + 2 * expansionPack
numDeserts = 1 + expansionPack


# In[4]:


numTwos = 1 + expansionPack
numThrees = 2 + expansionPack
numFours = 2 + expansionPack
numFives = 2 + expansionPack
numSixes = 2 + expansionPack
numEights = 2 + expansionPack
numNines = 2 + expansionPack
numTens = 2 + expansionPack
numElevens = 2 + expansionPack
numTwelves = 1 + expansionPack


# In[5]:


resources = ['wood'] * numWoods
resources += ['brick'] * numBricks
resources += ['sheep'] * numSheeps
resources += ['wheat'] * numWheats
resources += ['rock'] * numRocks

resources


# In[6]:


numbers = [2] * numTwos
numbers += [3] * numThrees
numbers += [4] * numFours
numbers += [5] * numFives
numbers += [6] * numSixes
numbers += [8] * numEights
numbers += [9] * numNines
numbers += [10] * numTens
numbers += [11] * numElevens
numbers += [12] * numTwelves

numbers


# In[7]:


len(resources) == len(numbers)


# In[8]:


random.shuffle(resources)
random.shuffle(numbers)


# In[9]:


robbers = [0]*len(resources)
rows = [0]*len(resources)


# In[10]:


tiles = pd.DataFrame(data = list(zip(resources, numbers, robbers, rows)), columns = ['resource', 'number', 'robber', 'row'])
tiles


# In[11]:


desertResource = ['desert'] * numDeserts
desertNumber = [None] * numDeserts
desertRobber = [0] * numDeserts
desertRows = [0] * numDeserts

deserts = pd.DataFrame(data = list(zip(desertResource, desertNumber, desertRobber, desertRows)), columns = ['resource', 'number', 'robber', 'row'])
tiles = pd.concat([tiles, deserts])
tiles = tiles.sample(frac=1).reset_index(drop=True)

tiles


# In[12]:


rowLen = 3
rowNum = 0
rowPos = 0
for res in range(len(tiles.resource)):
    tiles.iat[res, 3] = rowNum
    if rowPos == rowLen - 1:
        rowNum += 1
        rowPos = 0
        if res > len(tiles.resource)/2:
            rowLen -= 1
        else:
            rowLen += 1
    else:
        rowPos += 1
tiles


# In[13]:


resources = []
for i in range(max(tiles.row)+1):
    newResources = []
    for j in tiles.loc[tiles.row == i].index:
        newResources += [j]*2
    resources += [*newResources]*3
resources


# In[14]:


row = []
currentRow = 0
rowLengths = tiles.groupby(by='row').count().resource
newRow = []
for i in range(len(rowLengths)):
    newRow = []
    for j in range(3):
        newRow += [currentRow]*(2*rowLengths[i])
        currentRow += 1
    row += newRow
row


# In[15]:


col = []
rowLengths = tiles.groupby(by='row').count().resource
for i in range(len(rowLengths)):
    newCol = []
    if rowLengths[i] < max(rowLengths):
        newCol += [*range(max(rowLengths) - rowLengths[i], max(rowLengths) - rowLengths[i] + rowLengths[i]*2)]
    else:
        newCol += [*range(rowLengths[i]*2)]
    col += [*newCol]*3
col


# In[16]:


edgeNum = [0] * len(resources)
edgePair = [None] * len(resources)
vertexEdge1 = [0] * len(resources)
vertexEdge2 = [0] * len(resources)

edgeStates = [0] * len(resources)
edgeOwners = [None] * len(resources)
vertexStates = [0] * len(resources)
vertexOwners = [None] * len(resources)


# In[17]:


board = pd.DataFrame(list(zip(edgeNum, edgePair, vertexEdge1, vertexEdge2, edgeStates, edgeOwners, vertexStates, vertexOwners, resources,row,col)), columns = ['edgeNum', 'edgePair', 'vertexEdge1', 'vertexEdge2','edgeState','edgeOwner','vertexState','vertexOwner','resource','row','col'])
board.loc[:,'edgeNum'] = board.index
board.head(50)


# In[18]:


for resource in range(len(tiles.resource)):
    vertexEdges1 = []
    for col in board.loc[board.resource == resource].col.unique():
        newEdges = board.loc[(board.resource == resource) & (board.col == col)].edgeNum
        if col % 2 == 0:
            newEdges = newEdges.iloc[::-1]
        vertexEdges1 += [*newEdges]
    for e in range(len(vertexEdges1)):
        board.loc[board.edgeNum == vertexEdges1[e], 'vertexEdge1'] = vertexEdges1[(e+1)%len(vertexEdges1)]


# In[19]:


for resource in range(len(tiles.resource)):
    vertexEdges2 = []
    for col in board.loc[board.resource == resource].col.unique():
        newEdges = board.loc[(board.resource == resource) & (board.col == col)].edgeNum
        if col % 2 == 1:
            newEdges = newEdges.iloc[::-1]
        vertexEdges2 += [*newEdges]
    for e in range(len(vertexEdges2)):
        board.loc[board.edgeNum == vertexEdges2[e], 'vertexEdge2'] = vertexEdges2[(e+1)%len(vertexEdges2)]


# In[20]:


for edgeNum in board.edgeNum:
    resourceEdges = board.loc[board.loc[:,'resource'] == board.loc[edgeNum, 'resource']]
    if board.loc[edgeNum,'row'] == min(resourceEdges.loc[:,'row']):
        edgePair = board.loc[(board.row == board.loc[edgeNum,'row'] - 1) & (board.col == board.loc[edgeNum,'col'])].edgeNum
        if len(edgePair) != 0:
            board.loc[edgeNum,'edgePair'] = sum(edgePair)
    elif board.loc[edgeNum,'row'] == max(resourceEdges.loc[:,'row']):
        edgePair = board.loc[(board.row == board.loc[edgeNum,'row'] + 1) & (board.col == board.loc[edgeNum,'col'])].edgeNum
        if len(edgePair) != 0:
            board.loc[edgeNum,'edgePair'] = sum(edgePair)
    elif board.loc[edgeNum,'col'] == min(resourceEdges.loc[:,'col']):
        edgePair = board.loc[(board.row == board.loc[edgeNum,'row']) & (board.col == board.loc[edgeNum,'col'] - 1)].edgeNum
        if len(edgePair) != 0:
            board.loc[edgeNum,'edgePair'] = sum(edgePair)
    elif board.loc[edgeNum,'col'] == max(resourceEdges.loc[:,'col']):
        edgePair = board.loc[(board.row == board.loc[edgeNum,'row']) & (board.col == board.loc[edgeNum,'col'] + 1)].edgeNum
        if len(edgePair) != 0:
            board.loc[edgeNum,'edgePair'] = sum(edgePair)


# In[21]:


tiles.head(5)


# In[22]:


board.head(5)


# In[23]:


edgeStates = pd.DataFrame(np.array([[0,'emtpy'], [1,'road']]), columns=['edgeState', 'state'])
edgeStates


# In[24]:


vertexStates = pd.DataFrame(np.array([[0,'emtpy'], [1,'settlement'],[2,'city']]), columns=['vertexStates', 'state'])
vertexStates


# In[25]:


sns.scatterplot(data = board, x = 'col', y = 'row', style = 'resource')


# # Wallets and DeX

# In[26]:


numPlayers = 4
players = [pd.DataFrame(columns = ['Asset','Quantity'])] * (numPlayers+1)
players


# In[27]:


initialRoads = 2
initialSettlements = 2

numRoads = (15*numPlayers) - (initialRoads*numPlayers)
numSettlements = (5*numPlayers) - (initialSettlements*numPlayers)
numCities = 4*numPlayers
numWoods = 19+5*expansionPack
numBricks = 19+5*expansionPack
numWheats = 19+5*expansionPack
numSheep = 19+5*expansionPack
numRocks = 19+5*expansionPack
numKnights = 14+6*expansionPack
numMonopoly = 2+expansionPack
numRoadBuilding = 2+expansionPack
numYearOfPlenty = 2+expansionPack
numVictoryPoints = 5
numLongestRoad = 1
numLargestArmy = 1

initialBank = pd.DataFrame(np.array([['road',numRoads], ['settlement',numSettlements], ['city', numCities],['wood',numWoods],['brick',numBricks],['wheat',numWheats],['sheep',numSheep],['rock',numRocks],['knight',numKnights],['monopoly',numMonopoly],['road_building',numRoadBuilding],['year_of_plenty', numYearOfPlenty], ['victory_point', numVictoryPoints],['longest_road',numLongestRoad],['largest_army',numLargestArmy]]), columns=['Asset', 'Quantity'])
initialBank


# In[28]:


initialPlayers = pd.DataFrame(np.array([['road',initialRoads], ['settlement',initialSettlements]]), columns=['Asset', 'Quantity'])
initialPlayers


# In[29]:


players[0] = pd.concat([players[0], initialBank], ignore_index = True)
players[0]


# In[30]:


for i in range(1,numPlayers + 1):
    players[i] = pd.concat([players[i], initialPlayers], ignore_index = True)
players[numPlayers]


# In[31]:


victoryPoints = pd.DataFrame(np.array([['settlement',1], ['city', 2], ['victory_point', 1],['longest_road',2],['largest_army',2]]), columns=['Asset', 'Victory Points'])
victoryPoints


# In[32]:


road = pd.DataFrame(np.array([['brick',1], ['wood', 1]]), columns=['Asset', 'Quantity'])
settlement = pd.DataFrame(np.array([['brick',1], ['wood', 1], ['wheat','1'],['sheep',1]]), columns=['Asset', 'Quantity'])
city = pd.DataFrame(np.array([['wheat',2], ['rock', 3]]), columns=['Asset', 'Quantity'])
devCard = pd.DataFrame(np.array([['wheat',1], ['sheep', 1],['rock',1]]), columns=['Asset', 'Quantity'])

buildCard = pd.DataFrame(np.array([['road',road],['settlement',settlement], ['city', city], ['devCard',devCard]]), columns=['Asset', 'Resources'])
buildCard.set_index('Asset', inplace = True)
buildCard.loc['settlement'].Resources


# In[33]:


buildCard


# # Game Theory

# In[34]:


def countVP(player):
    joined = player.join(victoryPoints.set_index('Asset'), how = 'inner',on='Asset')
    VP = (joined.Quantity.astype(int)).dot(joined['Victory Points'].astype(int))
    if VP >= 10:
        print('Player has won with a total of '+str(VP)+' VP!')
        print(joined)
    return VP


# In[35]:


def getTileScores(player):
#     tileScores = []
#     for tile in tiles[tiles.robber == 0].index:
#         tileScore = 0
#         edges = board[board.resource == tile].index
#         for edge in edges:
#             tileScore += board[edge].vertexState
#             tileScores.append(tileScore)
    tileScores = []
    for tile in tiles[tiles.robber == 0].index:
        tileScore = random.randint(0,len(tiles[tiles.robber == 0])-1)
        tileScores.append(tileScore)
    return tileScores


# In[36]:


def getEdgeScores(player):
    edgeScores = []
    for edge in board:
        edgeScore = random.randint(0,len(board)-1)
        edgeScores.append(edgeScore)
    return edgeScores


# In[37]:


def getResourceScores():
    resourceScores = pd.DataFrame(data = {'wood':random.randint(0,4),'brick':random.randint(0,4),'wheat':random.randint(0,4),'sheep':random.randint(0,4),'rock':random.randint(0,4)})
    return resourceScores


# In[38]:


def robber(player):
    #target players who r over 6 VP
    maxVP = 6
    targetPlayers = []
    players_copy = players
    del players_copy[[0,players[player].index]]
    for i in players_copy.index:
        if countVP(players[i]) > maxVP:
            targetPlayers += i
    if len(targetPlayers) == 0:
        targetPlayers = players_copy
    
    #block the tile with the highest tile score (ie total resources paid out when rolled)
    tileScores = []
    for tile in tiles[tiles.robber == 0].index:
        tileScore = 0
        edges = board[board.resource == tile].index
        for edge in edges:
            if board.vertexOwner[edge] in targetPlayers:
                tileScore += board[edge].vertexState
                tileScores.append(tileScore)
    #remove tiles with player as owner
    blockedTile = tileScores.index(max(tileScores))
    tiles.assign(robber=0)
    tiles.loc[blockedTile, 'robber']  = 1
    
    #steal a random card from a player on that tile
    resource_board = board[board.resource == blockedTile]
    if len(resource_board.dropna()) != 0:
        if len(resource_board.dropna()) == 1 and sum(resource_board.vertexOwner) != player:
            targetPlayer = resource_board.vertexOwner
        else:
            VPs = []
            for owner in resource_board.loc[resource_board.vertexOwner != player, 'vertexOwner']:
                VPs.append(countVP(owner))
            targetPlayer = resource_board.vertexOwner[VPs.index(max(VPs))]
            
        resources = []
        for i in range(len(buildCard)):
            resources += [*buildCard.iloc[i].Resources.Asset]
        resources = np.unique(resources)
        resources = pd.DataFrame(data = resources, columns = ['Asset'])
        playerHand = players[sum(targetPlayer)].join(resources.set_index('Asset'), how = 'inner',on = 'Asset')
        stolenCard = random.choices(playerHand.Asset, weights=playerHand.Quantity, k=1)
        players[sum(targetPlayer)].loc[players[sum(targetPlayer)].Asset == str(stolenCard), 'Quantity'] = players[sum(targetPlayer)].loc[players[sum(targetPlayer)].Asset == str(stolenCard), 'Quantity'] - 1


# In[39]:


def rollNum(): 
    return random.randint(2,12)


# In[40]:


#def discardCards():
    #for each player, check strategy and determine most valuable resources
    #discard first floor(n/2) cards


# In[41]:


def collectResources(numRolled):
    rolledResources = tiles[(tiles.number == numRolled) & (tiles.robber == 0)].index
    for resource in rolledResources:
        resource_board = board[board.resource == resource]
        for i in resource_board.vertexOwner.index:
            owner = resource_board.vertexOwner[i]
            state = resource_board.vertexState[i]
            if owner == None:
                continue
            else:
                if players[owner].loc[players[owner].Asset == board.loc[resource, 'resource'], 'Quantity'].empty:
                    players[owner].loc[len(players[owner].index)] = [board.loc[resource, 'resource'], 1 + state]
                else: 
                    players[owner].loc[players[owner].Asset == board.loc[resource, 'resource'], 'Quantity'] = players[owner].loc[players[owner].Asset == board.loc[resource, 'resource'], 'Quantity'] + 1 + state


# In[42]:


def playDevCard(player,card):
    if card == 'knight':
        robber(player)
    elif card == 'monopoly':
        #identify most valuable resource
        #monopolize it: remove from all wallets and add to player
    elif card == 'road_building'
        buildRoad(player)
        buildRoad(player)
    elif card == 'year_of_plenty'
        #identify most valuable resources
        #take top two from bank


# In[ ]:


# def trade(player):
    #identify most valuable resources for player
    #get list of other players with desired resource 
    #get list of combinations of cards in hand worth less than resource 
    #propose trade to other player
    #other player evaluates trade and accepts/declines
    #if accepts, swap resources


# In[ ]:


# def buildRoad(player):
    #scan all available edges: accessible by road and available 
    #determine score for each of their corresponding resources for both left vertex and right vertex
    #swap resources for road
    #update board owners 


# In[ ]:


# def buildSettlement(player):
    #scan all available edges: accessible by road and no settlements around
    #determine score for each of their corresponding resources (ie probability * value of resource)
    #swap resources for settlement
    #update board owners 


# In[ ]:


# def buildCity(player):
    #scan all settlements 
    #determine score for each of their corresponding resources (ie probability * value of resource)
    #swap resources for city
    #update board owners 


# In[ ]:


# def buyDevCard(player):
    #select random devCard from bank
    #swap resources


# ## Strategy

# In[ ]:


# horizontalStrategy_weights = pd.DataFrame(np.array([['road',3/4], ['settlement', 1], ['city', 1],['knight',None],['year_of_plenty',None],['monopoly',None],['road_building',None],['victory_point',1],['wood',1/4],['brick',1/4],['wheat',1/4],['sheep',1/4],['rock',1/6],['longest_road',2],['largest_army',1],['devCard',1/2]]), columns=['Asset', 'Weights'])
# verticalStrategy_weights = pd.DataFrame(np.array([['road',None], ['settlement', 1], ['city', 1],['knight',None],['year_of_plenty',None],['monopoly',None],['road_building',None],['victory_point',1],['wood',None],['brick',None],['wheat',13/15],['sheep',2/3],['rock',13/15],['longest_road',None],['largest_army',2],['devCard',2/3]]), columns=['Asset', 'Weights'])
# strategies = pd.DataFrame(np.array([['Horizontal',horizontalStrategy_weights], ['Vertical', verticalStrategy_weights]]), columns=['Strategy', 'Weights'])
# strategies


# In[ ]:


# strategies.loc[0,'Weights']


# In[54]:


actions = ['Play Knight','Roll Num', 'Play Dev Card','Trade','Build Road','Build Settlement','Build City','Buy Dev Card']
weights0 = [random.randint(0,len(actions)-1),random.randint(0,len(actions)-1),random.randint(0,len(actions)-1),random.randint(0,len(actions)-1),random.randint(0,len(actions)-1),random.randint(0,len(actions)-1),random.randint(0,len(actions)-1),random.randint(0,len(actions)-1)]
strategy0 = pd.DataFrame(data = list(zip(actions,weights0)), columns = ['Action','Weight'])


# In[54]:


actions = ['Play Knight','Roll Num', 'Play Dev Card','Trade','Build Road','Build Settlement','Build City','Buy Dev Card']
weights1 = [random.randint(0,len(actions)-1),random.randint(0,len(actions)-1),random.randint(0,len(actions)-1),random.randint(0,len(actions)-1),random.randint(0,len(actions)-1),random.randint(0,len(actions)-1),random.randint(0,len(actions)-1),random.randint(0,len(actions)-1)]
weights2 = [random.randint(0,len(actions)-1),random.randint(0,len(actions)-1),random.randint(0,len(actions)-1),random.randint(0,len(actions)-1),random.randint(0,len(actions)-1),random.randint(0,len(actions)-1),random.randint(0,len(actions)-1),random.randint(0,len(actions)-1)]
strategy1 = pd.DataFrame(data = list(zip(actions,weights1)), columns = ['Action','Weight'])
strategy2 = pd.DataFrame(data = list(zip(actions,weights2)), columns = ['Action','Weight'])


# In[56]:


strategies = []
strategies.append(strategy0)
strategies.append(strategy1)
strategies.append(strategy2)
strategies


# In[57]:


strategies[1]


# In[60]:


playerStrategies = [None]*len(players)
playerStrategies[0] = strategies[0]
players_copy = players
del players_copy[0]
for player in players_copy.index:
    playerStrategies[player] = random.choice(strategies)
playerStrategies


# ## ML/AI

# In[3]:


# Run 10,000 games with random player strategies for 3 players, 4 players, 5 players, 6 players
# Record all board states, player states, player strategies, VP
# Add new strategy with all weights = 0
# Build 3 ML models on VP >= 10 for 3-4 players and 5-6 players

# Run 10,000 games with randomly chosen player strategies out of the 3 ML models
# Record all board states, player states, player strategies, VP
# Retrain 3 ML models
# Repeat


# ## Setup

# In[ ]:


#select random player to go first
#for each player:
    #scan all edges and their corresponding resources
    #determine score of all edges
    #place road
    #place settlement
#for each player in reverse order:
    #scan all edges and their corresponding resources
    #determine score of all edges
    #place road
    #place settlement


# ## Gameplay

# In[ ]:


winner = None
currentPlayer = #player who went first
availableActions = [*range(len(playerStrategies[currentPlayer]))]
while winner == None:
    action = random.choices(availableActions, weights=playerStrategies[currentPlayer].Weight, k=1)
    #if action == 0:
        playDevCard(currentPlayer, 'knight')
        del availableActions[['Play Knight','Play Dev Card']]
    #elif action == 1:
        numRolled = rollNum()
        if numRolled == 7:
            discardCards()
            robber(currentPlayer)
        else:
            collectResources(numRolled)
        del availableActions['Roll Num']
    #elif action == 2:
        playDevCard(currentPlayer, #devCard)
        del availableActions[['Play Knight','Play Dev Card']]
    #elif action == 3:
        trade(currentPlayer)
        del availableActions['Trade']
    #elif action == 4:
        buildRoad(player)
    #elif action == 5:
        buildSettlement(player)
    #elif action == 6:
        buildCity(player)
    #elif action == 7:
        buyDevCard(player)
    if countVP(currentPlayer) >= 10:
        winner = currentPlayer
    else:
        currentPlayer += 1%numPlayers
        availableActions = [*range(len(playerStrategies[currentPlayer]))]

