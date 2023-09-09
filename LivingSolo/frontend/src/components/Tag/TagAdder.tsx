import { styled } from "styled-components";
import { CondRendAnimState, condRendMounted, condRendUnmounted, onAnimEnd } from "../../utils/Rendering";
import { useState } from "react";
import { TagElement } from "../../store/slices/tag";
import { useDispatch, useSelector } from "react-redux";
import { TagInputForTodo } from "./TagInput";
import { AppDispatch } from "../../store";


const DEFAULT_OPTION = '$NONE$';

interface TodoAdderProps {
    addMode: CondRendAnimState,
    setAddMode: React.Dispatch<React.SetStateAction<CondRendAnimState>>,
};

interface TodoEditorProps extends TodoAdderProps {
    editObj: TagElement,
    editCompleteHandler: () => void,
};

const todoSkeleton = {
    name: '',
    category: DEFAULT_OPTION,
    priority: 0,
    deadline: '',
    is_hard_deadline: false,
    period: 0,
};

export const TagAdder = ({ addMode, setAddMode } : TodoAdderProps) => {
    const dispatch = useDispatch<AppDispatch>();

    return <TodoAdderWrapper style={addMode.isMounted ? condRendMounted : condRendUnmounted}
                            onAnimationEnd={() => onAnimEnd(addMode, setAddMode)}>
                                TAG ADDER
    </TodoAdderWrapper>
};

export const TagEditor = ({ addMode, setAddMode, editObj, editCompleteHandler } : TodoEditorProps) => {
    const dispatch = useDispatch<AppDispatch>();

    return <TodoAdderWrapper style={addMode.isMounted ? condRendMounted : condRendUnmounted}
                            onAnimationEnd={() => onAnimEnd(addMode, setAddMode)}>
                                Tag Editor
    </TodoAdderWrapper>
};

const TodoAdderWrapper = styled.div`
    width: 100%;
    display: flex;
    flex-direction: column;

    margin-bottom: 10px;
`;

const TodoAdder1stRow = styled.div`
    width: 100%;
    display: grid;
    grid-gap: 15px;
    grid-template-columns: 1fr 1fr 1fr 1fr 3fr;
    
    margin-bottom: 10px;
`;
const TodoEditor1stRow = styled.div`
    width: 100%;
    display: grid;
    grid-gap: 15px;
    grid-template-columns: 3fr 1fr 1fr 1fr 3fr 5fr;
    
    margin-bottom: 10px;

    input {
        max-width: 80px; // For Amount Input
        text-align: center;
    }
`;
const TodoAdderAddInputs = styled.div`
    display: flex;
    flex-direction: column;
    align-items: center;
    label {
        color: var(--ls-gray_google2);
        margin-bottom: 2px;
    }
`;
const TodoPeriodicLabel = styled.div`
    margin-bottom: 2px;

    input {
        margin-left: 10px;
    }
`;
const TodoAdder2ndRow = styled.div`
    width: 100%;
    padding: 10px 10px 15px 10px;
    border-bottom: 1.5px solid gray;
    margin-bottom: 10px;
    
    display: flex;
    align-items: center;
`;
const TodoAdder2ndRowInputWrapper = styled.div`
    width  : 100%;
    display: flex;
    justify-content: space-between;

    input {
        width: 100%;
        padding: 10px;
        margin-right: 20px;
    }
    button {
        padding: 10px;
    }
`;

const TodoElementColorCircle = styled.div<{ color: string, ishard: string }>`
    width: 20px;
    height: 20px;
    border-radius: 50%;
    border: ${props => ((props.ishard === 'true') ? '2px solid var(--ls-red)' : 'none')};
    background-color: ${props => (props.color)};;
    
    margin-right: 10px;

    display: flex;
    justify-content: center;
    align-items: center;

    cursor: pointer;
`;