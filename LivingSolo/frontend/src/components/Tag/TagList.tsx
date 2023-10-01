import { TagElement, deleteTagPreset, selectTag } from "../../store/slices/tag";
import { TagBubbleCompact } from "../general/TagBubble";
import { useDispatch, useSelector } from "react-redux";
import { TagClassAdder, TagClassAdderHeight, TagClassEditor } from "./TagClassAdder";
import { CondRendAnimState } from "../../utils/Rendering";
import styled from "styled-components";
import { TagViewerMode } from "../../containers/TagMain";
import { TagAdder, TagAdderHeight, TagEditor } from "./TagAdder";
import { TagPresetAdder, TagPresetAdderHeight, TagPresetEditor } from "./TagPresetAdder";
import { AppDispatch } from "../../store";
import { IPropsColor } from "../../utils/Interfaces";
import { getContrastYIQ } from "../../styles/color";

interface TagListProps {
    addMode: CondRendAnimState,
    setAddMode: React.Dispatch<React.SetStateAction<CondRendAnimState>>,
    tagViewerMode: TagViewerMode,
    setTagViewerMode: React.Dispatch<React.SetStateAction<TagViewerMode>>,
    tagSelectHandler: (tag : TagElement) => void,
};


// containers/TagMain.tsx에서 사용되는 TagList 패널.
export const TagList = ({ addMode, setAddMode, tagViewerMode, setTagViewerMode, tagSelectHandler } : TagListProps) => {
    const dispatch = useDispatch<AppDispatch>();
    const { elements, index, preset } = useSelector(selectTag);

    const classEditHandler = () => {

    };

    return <>
        {
              tagViewerMode === TagViewerMode.TagClass && <TagClassListPosition>
                {addMode.showElem && ( true ? 
                  (<TagClassAdder addMode={addMode} setAddMode={setAddMode} />)
                :
                  (elements[0] && <TagClassEditor addMode={addMode} setAddMode={setAddMode} editObj={elements[0]} editCompleteHandler={classEditHandler}/>)
                )}
                <TagClassList style={addMode.showElem && addMode.isMounted ? { transform: `translateY(${TagClassAdderHeight})` } : { transform: "translateY(0px)" }}>
                  {elements.map((tagClass) => {
                    return <TagClassListElement key={tagClass.id}>
                      <TagClassListElementHeader color={tagClass.color}>{tagClass.name}</TagClassListElementHeader>
                      <div>
                        {tagClass.tags?.map((tag) => {
                          return <TagBubbleCompact color={tag.color} key={tag.id} onClick={() => tagSelectHandler(tag)}>{tag.name}</TagBubbleCompact>
                        })}
                      </div>
                    </TagClassListElement>
                  })}
                </TagClassList>         
              </TagClassListPosition>
            }
            {
              tagViewerMode === TagViewerMode.Tag && <TagClassListPosition>
                {addMode.showElem && ( true ? 
                  (<TagAdder addMode={addMode} setAddMode={setAddMode} />)
                :
                  (elements[0] && <TagEditor addMode={addMode} setAddMode={setAddMode} editObj={elements[0]} editCompleteHandler={classEditHandler}/>)
                )}
                <TagLists style={addMode.showElem && addMode.isMounted ? { transform: `translateY(${TagAdderHeight})` } : { transform: "translateY(0px)" }}>
                  {index.map((tag) => {
                    return <TagBubbleCompact color={tag.color} key={tag.id} onClick={() => tagSelectHandler(tag)}>{tag.name}</TagBubbleCompact>
                  })}     
                </TagLists>         
              </TagClassListPosition>
            }
            {
              tagViewerMode === TagViewerMode.TagPreset && <TagClassListPosition>
                {addMode.showElem && ( true ? 
                  (<TagPresetAdder addMode={addMode} setAddMode={setAddMode} />)
                :
                  (elements[0] && <TagPresetEditor addMode={addMode} setAddMode={setAddMode} editObj={elements[0]} editCompleteHandler={classEditHandler}/>)
                )}
                <TagPresetList style={addMode.showElem && addMode.isMounted ? { transform: `translateY(${TagPresetAdderHeight})` } : { transform: "translateY(0px)" }}>
                  {preset.map((preset) => {
                    return <TagPresetListElement key={preset.id}>
                      <span>{preset.name}</span>
                      <div>
                        {preset.tags.map((tag) => <TagBubbleCompact color={tag.color} key={tag.id} onClick={() => tagSelectHandler(tag)}>{tag.name}</TagBubbleCompact>)}
                      </div>
                      <span onClick={() => dispatch(deleteTagPreset({id: preset.id}))}>삭제</span>
                    </TagPresetListElement>
                  })} 
                </TagPresetList>         
              </TagClassListPosition>
            }
    </>
}

const TagClassListPosition = styled.div`
  width: 100%;
  position: relative;
`;

const TagClassList = styled.div`
  margin-bottom: 20px;

  width: 100%;
  position: absolute;
  top: 0px;

  transition-property: all;
  transition-duration: 250ms;
  transition-delay: 0s;
`;

const TagLists = styled.div`
  margin-top: 10px;
  margin-bottom: 20px;

  position: absolute;
  top: 0px;

  transition-property: all;
  transition-duration: 250ms;
  transition-delay: 0s;
`;

const TagPresetList = styled.div`
  margin-top: 10px;
  margin-bottom: 20px;

  position: absolute;
  top: 0px;

  transition-property: all;
  transition-duration: 250ms;
  transition-delay: 0s;

  width: 100%;
`;

const TagPresetListElement = styled.div`
  padding: 10px;
  border-bottom: 1px solid gray;

  display: grid;
  grid-template-columns: 4fr 10fr;
  align-items: center;
  span {

  }
`;

const TagClassListElement = styled.div`
  margin-bottom: 10px;
  padding-bottom: 10px;
  padding-left: 10px;
  border-bottom: 1px solid black;

  &:first-child{
    padding-top: 10px;
  }
`;

const TagClassListElementHeader = styled.div<IPropsColor>`
  padding: 6px 10px 6px 10px;
  width: 200px;
  border-radius: 10px;
  text-align: center;

  margin-bottom: 4px;
  ${({ color }) =>
    color &&
    `
      background: ${color};
      color: ${getContrastYIQ(color)}
    `}
`;