package org.ppoplib.ppo.env;

public class Action {
    public ActionType type;
    private String name;

    public Action(){
        
    }

    public enum ActionType {
        DISCRETE,
        CONTINUOUS
    }
}
