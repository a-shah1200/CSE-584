class Policy(torch.nn.Module):
  def __init__(self,state,action):
    super().__init__()
    # Three - layer neural network
    self.input=Linear(state,128) # First layer maps state space to hidden layer
    self.dense=Linear(128,128) # Hidden layer for feature extraction
    self.output=Linear(128,action) # Output layer produces Q - values for each action

  def forward(self,tensor):
    x=F.relu(self.input(tensor)) # Apply ReLU to first layer output
    x=F.relu(self.dense(x)) # Apply ReLU to hidden layer output
    return self.output(x) # Return raw Q - values


class memory():
  def __init__(self,capacity):
    # Initialize circular buffer with fixed capacity
    self.mem=deque([],maxlen=capacity)

  def add(self,state,action,reward,next_state):
    # Store transition tuple in memory
    self.mem.append((state,action,reward,next_state))

  def get_sample(self,size):
    # Randomly sample batch of a particular size of transitions for training
    return random.sample(self.mem,size)

  def __len__(self):
    return len(self.mem)


class DQN():
  def __init__(self,env,PolicyClass):
    self.env=env
    self.LR=1e-4
    self.BATCH_SIZE=128
    self.theta=0.005
    self.START=0.9
    self.END=0.05
    self.GAMMA=0.99
    self.NUM_EPISODES=600 if torch.cuda.is_available() else 400
    self.DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.CAPACITY=10000
    self.DECAY=1000
    self.steps=0
    self.Policy_net,self.Target_net,self.optimizer,self.buffer=self.initialize(self.env,PolicyClass)
    self.Policy_net.to(self.DEVICE)
    self.Target_net.to(self.DEVICE)
    self.Target_net.load_state_dict(self.Policy_net.state_dict())
    self.avg_reward=[]

  def initialize(self,env,PolicyClass):
    n_actions=self.env.action_space.n
    state,_=self.env.reset()
    statesize=len(state)
    Policy_net=PolicyClass(statesize,n_actions).to(self.DEVICE)
    Target_net=PolicyClass(statesize,n_actions).to(self.DEVICE)
    Target_net.load_state_dict(Policy_net.state_dict())
    optimizer=torch.optim.AdamW(Policy_net.parameters(),lr=self.LR)
    buffer=memory(self.CAPACITY)
    return Policy_net,Target_net,optimizer,buffer

  def soft_update(self):
    # Get current weights of both networks
    w_1=self.Policy_net.state_dict()
    w_2=self.Target_net.state_dict()
    # Perform soft update .
    for key in w_1:
      # Update formula : theta_target = theta * policy + (1 - theta ) * target
      w_2[key]=self.theta*w_1[key]+(1-self.theta)*w_2[key]
    # Load updated weights into target network
    self.Target_net.load_state_dict(w_2)

  def choose_action(self,state):
    self.steps+=1
    # Calculate exploration probability using exponential decay
    threshold = self.END + (self.START - self.END) * math.exp(-1. * self.steps / self.DECAY)
    sample=random.random()
    if sample>threshold:
      # Exploit : choose action with highest Q - value
      with torch.no_grad():
        return self.Policy_net(state).max(1)[1].view(1, 1)
    else:
      # Explore : choose random action
      return torch.tensor([[self.env.action_space.sample()]], device=self.DEVICE, dtype=torch.long)

  def train(self):
    # Skip if not enough samples in buffer
    if len(self.buffer)<self.BATCH_SIZE:
      return None

    # Sample random batch from replay buffer
    batch=self.buffer.get_sample(self.BATCH_SIZE)
    # Create mask for non - final states
    non_final_index=torch.tensor(tuple([True if i[3]!=None else False for i in batch]),device=self.DEVICE,dtype=torch.bool)
    non_final_states=torch.cat([i[3] for i in batch if i[3]!=None])
    # Prepare batch data
    batch=list(zip(*batch))
    state_batch=torch.cat(batch[0])     # Current states
    action_batch=torch.cat(batch[1])    # Actions taken
    reward_batch=torch.cat(batch[2])    # Rewards received
    
    # Get predicted Q - values using policy network
    predicted_value=self.Policy_net(state_batch).gather(1,action_batch)
    temp=torch.zeros(self.BATCH_SIZE).to(self.DEVICE)

    # Calculate target Q - values using target network
    with torch.no_grad():
      temp[non_final_index]=self.Target_net(non_final_states).max(1)[0]
    
    # Compute target values using Bellman equation
    target_value=reward_batch+self.GAMMA*temp

    # Calculate loss
    criteria=torch.nn.SmoothL1Loss()
    loss=criteria(predicted_value,target_value.unsqueeze(1))

    # Update the policy network
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

  def loop(self):
    max_ep_len = 500   # maximum length of episode
    max_training_timesteps = int(1e5)  # maximum number of training timesteps
    save_freq=400*2  # save frequency
    log_running_reward = 0
    log_running_episodes = 0

    while self.steps <= max_training_timesteps:
      state,_=self.env.reset()
      state=torch.tensor(state,dtype=torch.float32,device=self.DEVICE).unsqueeze(0)
      total_reward=0
      for t in range(1, max_ep_len+1):
        action=self.choose_action(state) # get action
        observation,reward,terminated,truncated,_=self.env.step(action.item())
        done=terminated or truncated # execute action to get next state and reward
        total_reward+=reward
        reward=torch.tensor([reward],device=self.DEVICE)
        if terminated:
          next_state=None
        else:
          next_state=torch.tensor(observation,dtype=torch.float32,device=self.DEVICE).unsqueeze(0)
        self.buffer.add(state,action,reward,next_state) # add the ( state , action , reward , next_state )
        state=next_state
        self.train() # start training model
        self.soft_update() # update the model
        if self.steps%save_freq==0: # Save the average rewards ( For plotting purposes )
          log_avg_reward = log_running_reward / log_running_episodes
          log_avg_reward = round(log_avg_reward, 4)
          self.avg_reward.append((self.steps, log_avg_reward))
          log_running_reward = 0
          log_running_episodes = 0
        if done: # if episode is terminated we dont need to go to max_ep_len
          break
      log_running_reward += total_reward
      log_running_episodes += 1

  def plot(self,name):
    import matplotlib.pyplot as plt
    plt.xlabel("Time steps")
    plt.ylabel("Avg episodic reward")
    plt.title(f"DQN plot for {name}")
    plt.plot([i[0] for i in self.avg_reward],[i[1] for i in self.avg_reward] )
    plt.show()


env = gym.make("CartPole-v1")
cart=DQN(env,Policy)
cart.loop()
cart.plot("CartPole-V1")



