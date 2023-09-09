import { styled } from "styled-components";
import { CondRendAnimState, condRendMounted, condRendUnmounted, onAnimEnd } from "../../utils/Rendering";
import { useState } from "react";
import { TagElement, createTag, selectTag } from "../../store/slices/tag";
import { useDispatch, useSelector } from "react-redux";
import { AppDispatch } from "../../store";
import { ColorCircleLarge, DEFAULT_COLOR, getRandomHex } from "../../styles/color";
import { ColorDialog } from "../general/ColorDialog";
import { DEFAULT_OPTION } from "../../utils/Constants";

export const TagAdderHeight = '70px';

interface TagAdderProps {
  addMode: CondRendAnimState,
  setAddMode: React.Dispatch<React.SetStateAction<CondRendAnimState>>,
};

interface TagEditorProps extends TagAdderProps {
  editObj: TagElement,
  editCompleteHandler: () => void,
};

const tagSkeleton = {
  name: '',
  color: DEFAULT_COLOR,
  period: 0,
};

export const TagAdder = ({ addMode, setAddMode } : TagAdderProps) => {
  const dispatch = useDispatch<AppDispatch>();

  const { elements } = useSelector(selectTag);
  const [tagName, setTagName] = useState<string>('');
  const [tagClass, setTagClass] = useState<string>(DEFAULT_OPTION);
  // Color Dialog
  const [open, setOpen] = useState<boolean>(false);
  const [color, setColor] = useState<string>(tagSkeleton.color);

  const handleClose = () => {
    setOpen(false);
  };
  const colorDialogOpenHandler = () => {
    setOpen(true);
  };

  const tagClassChangeHandler = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const target: string = e.target.value;
    const prevClass = (tagClass === DEFAULT_OPTION) ? undefined :  elements.find((e) => (e.id.toString() === tagClass));
    const targetClass = elements.find((e) => (e.id.toString() === target));

    if(targetClass === undefined){
      console.log("Critical Error from tagClassChangeHandler");
      return;
    }
    
    if(
      (tagClass === DEFAULT_OPTION && color !== DEFAULT_COLOR) ||
      (prevClass !== undefined && prevClass.color !== color)
    ){
      if(window.confirm('현재 색상이 기본값과 다릅니다. 색상을 선택한 태그 클래스의 색상으로 변경하시겠습니까?')){
        setColor(targetClass.color);
      }
    }else{
      setColor(targetClass.color);
    }

    setTagClass(targetClass.id.toString());
  };

  return <TagAdderWrapper style={addMode.isMounted ? condRendMounted : condRendUnmounted} onAnimationEnd={() => onAnimEnd(addMode, setAddMode)}>
    <TagAdderRow>
      <select data-testid="tagSelect" value={tagClass} onChange={tagClassChangeHandler}>
        <option disabled value={DEFAULT_OPTION}>
          - 태그 클래스 -
        </option>
        {elements.map(tagClass => {
          return (
              <option value={tagClass.id} key={tagClass.id}>
                {tagClass.name}
              </option>
          );
        })}
      </select>
      <ColorCircleLarge color={color}>
        <div onClick={() => { setColor(getRandomHex()); }}></div>
        <div className="clickable" onClick={colorDialogOpenHandler}>...</div>
      </ColorCircleLarge>
      <input type="text" placeholder='Tag  Name' value={tagName} onChange={(e) => setTagName(e.target.value)}/>
      <button disabled={tagName === '' || tagClass === DEFAULT_OPTION} onClick={() => { 
        dispatch(createTag({name: tagName, color, class: tagClass }));
        setTagName("");
        setColor(DEFAULT_COLOR);
      }}>Create</button>
  </TagAdderRow>
      <ColorDialog open={open} handleClose={handleClose} color={color} setColor={setColor}/>
  </TagAdderWrapper>
};

export const TagEditor = ({ addMode, setAddMode, editObj, editCompleteHandler } : TagEditorProps) => {
  const dispatch = useDispatch<AppDispatch>();

  return <TagAdderWrapper style={addMode.isMounted ? condRendMounted : condRendUnmounted}
                          onAnimationEnd={() => onAnimEnd(addMode, setAddMode)}>
                              Tag Editor
  </TagAdderWrapper>
};

const TagAdderWrapper = styled.div`
  width: 100%;
  height: 60px;

  display: flex;
  flex-direction: column;

  margin-bottom: 10px;
`;

const TagAdderRow = styled.div`
  display: grid;
  grid-template-columns: 2fr 3fr 15fr 2.5fr;
  align-items: center;

  padding: 10px 4px 10px 4px;
  border-bottom: 1.5px solid gray;

  input {
      padding: 10px;
      margin: 0px 15px 0px 10px;
      border: 1px solid gray;
      border-radius: 10px;
  }
  button {
      padding: 10px;
  }
`;