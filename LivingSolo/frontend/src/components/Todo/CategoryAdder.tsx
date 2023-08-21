import { styled } from "styled-components";
import { CondRendAnimState, condRendMounted, condRendUnmounted, onAnimEnd } from "../../utils/Rendering";
import { useEffect, useState } from "react";
import { TagElement } from "../../store/slices/tag";
import { TodoCategory, TodoCategoryCreateReqType, TodoCategoryEditReqType, createTodoCategory, editTodoCategory } from "../../store/slices/todo";
import { useDispatch } from "react-redux";
import { TagInputForTodoCategory } from "../Tag/TagInput";
import { AppDispatch } from "../../store";
import { getRandomHex } from "../../styles/color";


interface CategoryAdderProps {
    addMode: CondRendAnimState,
    setAddMode: React.Dispatch<React.SetStateAction<CondRendAnimState>>,
};

interface CategoryEditorProps extends CategoryAdderProps {
    editObj: TodoCategory,
    editCompleteHandler: () => void,
};

const todoCategorySkeleton = {
    name: '',
    color: '#000000',
}

export const CategoryAdder = ({ addMode, setAddMode } : CategoryAdderProps) => {
    const dispatch = useDispatch<AppDispatch>();

    // Category List - Create
    const [categTags, setCategTags] = useState<TagElement[]>([]);
    const [newTodoCategory, setNewTodoCategory] = useState<TodoCategoryCreateReqType>({...todoCategorySkeleton, tag: categTags});

    return <CategoryAdderWrapper style={addMode.isMounted ? condRendMounted : condRendUnmounted} onAnimationEnd={() => onAnimEnd(addMode, setAddMode)}>
    <CategoryAdderRow>
        <TodoElementColorCircle color={newTodoCategory.color} ishard={'false'} onClick={() => setNewTodoCategory((nTC) => { return {...nTC, color: getRandomHex()}})}></TodoElementColorCircle>
        <TagInputForTodoCategory tags={categTags} setTags={setCategTags} closeHandler={() => {}}/>
        <CategoryAdderInputWrapper>
            <input type="text" placeholder='Category Name' value={newTodoCategory.name} onChange={(e) => setNewTodoCategory((nTC) => { return {...nTC, name: e.target.value}})}/>
            <button onClick={() => { 
                dispatch(createTodoCategory({...newTodoCategory, tag: categTags}));
                setCategTags([]);
                setNewTodoCategory({...todoCategorySkeleton, tag: categTags});
            }}>Create</button>
        </CategoryAdderInputWrapper>
    </CategoryAdderRow>
    
</CategoryAdderWrapper>
};

export const CategoryEditor = ({ addMode, setAddMode, editObj, editCompleteHandler } : CategoryEditorProps) => {
    const dispatch = useDispatch<AppDispatch>();

    useEffect(() => {
        setCategTags(editObj.tag);
        setEditCategory({...editObj});
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [editObj.id]);

    // Category List - Edit
    const [categTags, setCategTags] = useState<TagElement[]>(editObj.tag);
    const [editCategory, setEditCategory] = useState<TodoCategoryEditReqType>({...editObj});

    return <CategoryAdderWrapper style={addMode.isMounted ? condRendMounted : condRendUnmounted} onAnimationEnd={() => onAnimEnd(addMode, setAddMode)}>
        <CategoryAdderRow>
            <TodoElementColorCircle color={editCategory.color} ishard={'false'} onClick={() => setEditCategory((eC) => { return {...eC, color: getRandomHex()}})}></TodoElementColorCircle>
            <TagInputForTodoCategory tags={categTags} setTags={setCategTags} closeHandler={() => {}}/>
            <CategoryAdderInputWrapper>
                <input type="text" placeholder='Category Name' value={editCategory.name} onChange={(e) => setEditCategory((eC) => { return {...eC, name: e.target.value}})}/>
                <button onClick={() => { 
                    dispatch(editTodoCategory({...editCategory, tag: categTags}));
                    editCompleteHandler();
                }}>Edit</button>
            </CategoryAdderInputWrapper>
        </CategoryAdderRow>
    </CategoryAdderWrapper>
};

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

const CategoryAdderWrapper = styled.div`
    width: 100%;
    display: flex;
    flex-direction: column;

    margin-bottom: 10px;
`;

const CategoryAdderRow = styled.div`
    display: grid;
    grid-template-columns: 1fr 5fr 13fr;
    align-items: center;

    padding: 4px;
    padding-bottom: 10px;
    border-bottom: 1.5px solid gray;
`;

const CategoryAdderInputWrapper = styled.div`
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