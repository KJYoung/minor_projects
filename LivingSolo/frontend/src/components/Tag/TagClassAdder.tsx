import { styled } from "styled-components";
import { CondRendAnimState, condRendMounted, condRendUnmounted, onAnimEnd } from "../../utils/Rendering";
import { useState } from "react";
import { TagElement, createTagClass } from "../../store/slices/tag";
import { useDispatch } from "react-redux";
import { AppDispatch } from "../../store";
import { ColorCircleLarge, DEFAULT_COLOR, getRandomHex } from "../../styles/color";
import { ColorDialog } from "../general/ColorDialog";

export const TagClassAdderHeight = '70px';

interface TagClassAdderProps {
    addMode: CondRendAnimState,
    setAddMode: React.Dispatch<React.SetStateAction<CondRendAnimState>>,
};

interface TagClassEditorProps extends TagClassAdderProps {
    editObj: TagElement,
    editCompleteHandler: () => void,
};

const tagSkeleton = {
    name: '',
    color: DEFAULT_COLOR,
    period: 0,
};

export const TagClassAdder = ({ addMode, setAddMode } : TagClassAdderProps) => {
    const dispatch = useDispatch<AppDispatch>();

    const [tagClassName, setTagClassName] = useState<string>('');
    // Color Dialog
    const [open, setOpen] = useState<boolean>(false);
    const [color, setColor] = useState<string>(tagSkeleton.color);

    const handleClose = () => {
        setOpen(false);
    };
    const colorDialogOpenHandler = () => {
        setOpen(true);
    };

    return <TagClassAdderWrapper style={addMode.isMounted ? condRendMounted : condRendUnmounted} onAnimationEnd={() => onAnimEnd(addMode, setAddMode)}>
        <TagAdderRow>
            <ColorCircleLarge color={color}>
                <div onClick={() => { setColor(getRandomHex()); }}></div>
                <div className="clickable" onClick={colorDialogOpenHandler}>...</div>
            </ColorCircleLarge>
            <input type="text" placeholder='Tag Class Name' value={tagClassName} onChange={(e) => setTagClassName(e.target.value)}/>
            <button onClick={() => { 
                dispatch(createTagClass({name: tagClassName, color }));
                setTagClassName("");
                setColor(DEFAULT_COLOR);
            }}>Create</button>
    </TagAdderRow>
        <ColorDialog open={open} handleClose={handleClose} color={color} setColor={setColor}/>
    </TagClassAdderWrapper>
};

export const TagClassEditor = ({ addMode, setAddMode, editObj, editCompleteHandler } : TagClassEditorProps) => {
    const dispatch = useDispatch<AppDispatch>();

    return <TagClassAdderWrapper style={addMode.isMounted ? condRendMounted : condRendUnmounted}
                            onAnimationEnd={() => onAnimEnd(addMode, setAddMode)}>
                                Tag Editor
    </TagClassAdderWrapper>
};

const TagClassAdderWrapper = styled.div`
    width: 100%;
    height: 60px;

    display: flex;
    flex-direction: column;

    margin-bottom: 10px;
`;

const TagAdderRow = styled.div`
    display: grid;
    grid-template-columns: 2fr 25fr 2fr;
    align-items: center;

    padding: 10px 4px 10px 4px;
    border-bottom: 1.5px solid gray;

    input {
        padding: 10px;
        margin-right: 15px;
        border: 1px solid gray;
        border-radius: 10px;
    }
    button {
        padding: 10px;
    }
`;