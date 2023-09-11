import { styled } from "styled-components";
import { CondRendAnimState, condRendMounted, condRendUnmounted, onAnimEnd } from "../../utils/Rendering";
import { useState } from "react";
import { TagElement, createTagPreset } from "../../store/slices/tag";
import { useDispatch } from "react-redux";
import { AppDispatch } from "../../store";
import { TagInputForTagPreset } from "./TagInput";

export const TagPresetAdderHeight = '70px';

interface TagPresetAdderProps {
  addMode: CondRendAnimState,
  setAddMode: React.Dispatch<React.SetStateAction<CondRendAnimState>>,
};

interface TagEditorProps extends TagPresetAdderProps {
  editObj: TagElement,
  editCompleteHandler: () => void,
};


export const TagPresetAdder = ({ addMode, setAddMode } : TagPresetAdderProps) => {
  const dispatch = useDispatch<AppDispatch>();

  const [tagName, setTagName] = useState<string>('');
  const [tags, setTags] = useState<TagElement[]>([]);

  return <TagPresetAdderWrapper style={addMode.isMounted ? condRendMounted : condRendUnmounted} onAnimationEnd={() => onAnimEnd(addMode, setAddMode)}>
    <TagPresetAdderRow>
      <TagInputForTagPreset tags={tags} setTags={setTags}/>
      <input type="text" placeholder='Tag  Name' value={tagName} onChange={(e) => setTagName(e.target.value)}/>
      <button disabled={tagName === '' || tags.length === 0} onClick={() => { 
        dispatch(createTagPreset({name: tagName, tags }));
        setTagName("");
        setTags([]);
      }}>Create</button>
  </TagPresetAdderRow>
  </TagPresetAdderWrapper>
};

export const TagPresetEditor = ({ addMode, setAddMode, editObj, editCompleteHandler } : TagEditorProps) => {
  const dispatch = useDispatch<AppDispatch>();

  return <TagPresetAdderWrapper style={addMode.isMounted ? condRendMounted : condRendUnmounted}
                          onAnimationEnd={() => onAnimEnd(addMode, setAddMode)}>
                              Tag Editor
  </TagPresetAdderWrapper>
};

const TagPresetAdderWrapper = styled.div`
  width: 100%;
  height: 60px;

  display: flex;
  flex-direction: column;

  margin-bottom: 10px;
`;

const TagPresetAdderRow = styled.div`
  display: grid;
  grid-template-columns: 10fr 8fr 2fr;
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