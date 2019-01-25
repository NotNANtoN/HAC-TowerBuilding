"""
"run_HAC.py" executes the training schedule for the agent.
 By default, the agent will alternate between exploration and testing phases.
 The number of episodes in the exploration phase can be configured in section 3 of "design_agent_and_env.py" file.
 If the user prefers to only explore or only test,
 the user can enter the command-line options ""--train_only" or "--test", respectively.
 The full list of command-line options is available in the "options.py" file.
"""
from comet_ml import Experiment
import csv
import pickle as cpickle
import agent as Agent
from utils import print_summary

NUM_BATCH = 10000
TEST_FREQ = 2

num_test_episodes = 100

success_list = []

def run_HAC(FLAGS,env,agent):
    experiment = Experiment(api_key="M03EcOc9o9kiG95hws4mq1uqI",
                        project_name="HAC", workspace="antonwiehe")

    # Print task summary
    print_summary(FLAGS,env)
    
    # Determine training mode.  If not testing and not solely training, interleave training and testing to track progress
    mix_train_test = False
    if not FLAGS.test and not FLAGS.train_only:
        mix_train_test = True
     
    for batch in range(NUM_BATCH):

        num_episodes = agent.other_params["num_exploration_episodes"]
        
        # Evaluate policy every TEST_FREQ batches if interleaving training and testing
        if mix_train_test and batch % TEST_FREQ == 0:
            print("\n--- TESTING ---")
            agent.FLAGS.test = True
            num_episodes = num_test_episodes            

            # Reset successful episode counter
            successful_episodes = 0

        for episode in range(num_episodes):
            
            print("\nBatch %d, Episode %d" % (batch, episode))
            
            # Train for an episode
            success = agent.train(env, episode)

            if success:
                print("Batch %d, Episode %d End Goal Achieved\n" % (batch, episode))
                
                # Increment successful episode counter if applicable
                if mix_train_test and batch % TEST_FREQ == 0:
                    successful_episodes += 1            

        # Save agent
        agent.save_model(episode)


        # Finish evaluating policy if tested prior batch
        if mix_train_test and batch % TEST_FREQ == 0:

            # Log performance
            success_rate = successful_episodes / num_test_episodes * 100
            print("\nTesting Success Rate %.2f%%" % success_rate)
            agent.log_performance(success_rate)
            agent.FLAGS.test = False

            experiment.set_step(batch)
            experiment.log_metric("Success rate", success_rate)
            success_list.append(success_rate)
            with open("successRates.csv", 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(success_list)

            if success_rate > 95:
                print("Success rate over 95\%!")
                break

            print("\n--- END TESTING ---\n")

            

    
    

     
