/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useEffect, useState } from 'react';
import styled from 'styled-components';
import { useDispatch, useSelector } from 'react-redux';
import { AppDispatch } from '../store';
import { TagElement, fetchTagPresets, fetchTags, fetchTagsIndex, selectTag } from '../store/slices/tag';
import { TagBubbleCompact } from '../components/general/TagBubble';
import { IPropsActive, IPropsColor } from '../utils/Interfaces';
import { getContrastYIQ } from '../styles/color';
import { CondRendAnimState, defaultCondRendAnimState, toggleCondRendAnimState } from '../utils/Rendering';
import { TagClassAdder, TagClassAdderHeight, TagClassEditor } from '../components/Tag/TagClassAdder';
import { TagAdder, TagAdderHeight, TagEditor } from '../components/Tag/TagAdder';
import { TagPresetAdder, TagPresetAdderHeight, TagPresetEditor } from '../components/Tag/TagPresetAdder';

enum TagViewerMode {
  TagClass, Tag, TagPreset
};

const TagMain = () => {
  const [addMode, setAddMode] = useState<CondRendAnimState>(defaultCondRendAnimState);
  const [tagViewerMode, setTagViewerMode] = useState<TagViewerMode>(TagViewerMode.TagClass);
  const [selectedTag, setSelectedTag] = useState<TagElement | undefined>();

  const dispatch = useDispatch<AppDispatch>();

  const { elements, index, preset, errorState } = useSelector(selectTag);

  // Fetch Tag Related Things!
  useEffect(() => {
      dispatch(fetchTags());
      dispatch(fetchTagsIndex());
      dispatch(fetchTagPresets());
    }, [dispatch, errorState]);

  const classEditHandler = () => {

  };

  const classAddToggleHandler = () => {
    toggleCondRendAnimState(addMode, setAddMode); // ON
  };

  const tabChangeHandler = (newTabState : TagViewerMode) => {
    if(newTabState === tagViewerMode)
      return;
    if(addMode.showElem)
      classAddToggleHandler();
    setTagViewerMode(newTabState);
  };

  const tagSelectHandler = (tag: TagElement) => {
    setSelectedTag(tag);
  };

  return (
    <Wrapper>
      <InnerWrapper>
        <LeftWrapper>
          <ListWrapper>
            <TagTabHeader>
              <div>
                <TagTabName active={(tagViewerMode === TagViewerMode.TagClass).toString()} onClick={() => tabChangeHandler(TagViewerMode.TagClass)}>TagClass</TagTabName>
                <TagTabName active={(tagViewerMode === TagViewerMode.Tag).toString()}  onClick={() => tabChangeHandler(TagViewerMode.Tag)}>Tag</TagTabName>
                <TagTabName active={(tagViewerMode === TagViewerMode.TagPreset).toString()} onClick={() => tabChangeHandler(TagViewerMode.TagPreset)}>TagPreset</TagTabName>
              </div>
              <div>
                <span className='clickable' onClick={classAddToggleHandler}>+</span>
              </div>
            </TagTabHeader>
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
                <TagList style={addMode.showElem && addMode.isMounted ? { transform: `translateY(${TagAdderHeight})` } : { transform: "translateY(0px)" }}>
                  {index.map((tag) => {
                    return <TagBubbleCompact color={tag.color} key={tag.id} onClick={() => tagSelectHandler(tag)}>{tag.name}</TagBubbleCompact>
                  })}     
                </TagList>         
              </TagClassListPosition>
            }
            {
              tagViewerMode === TagViewerMode.TagPreset && <TagClassListPosition>
                {addMode.showElem && ( true ? 
                  (<TagPresetAdder addMode={addMode} setAddMode={setAddMode} />)
                :
                  (elements[0] && <TagPresetEditor addMode={addMode} setAddMode={setAddMode} editObj={elements[0]} editCompleteHandler={classEditHandler}/>)
                )}
                <TagList style={addMode.showElem && addMode.isMounted ? { transform: `translateY(${TagPresetAdderHeight})` } : { transform: "translateY(0px)" }}>
                  {preset.map((preset) => {
                    return <div key={preset.id}>
                        {preset.name}
                        {preset.tags.map((tag) => <TagBubbleCompact color={tag.color} key={tag.id} onClick={() => tagSelectHandler(tag)}>{tag.name}</TagBubbleCompact>)}
                    </div>
                  })} 
                </TagList>         
              </TagClassListPosition>
            }
          </ListWrapper>
        </LeftWrapper>
        <RightWrapper>
          {selectedTag && <TagBubbleCompact color={selectedTag.color}>{selectedTag.name}</TagBubbleCompact>}
        </RightWrapper>
      </InnerWrapper>
    </Wrapper>
  );
};

export default TagMain;

const Wrapper = styled.div`
  width: 100%;
  height: 100%;
  min-height: 100vh;
  background-color: #ffffff;
  display: flex;
  flex-direction: column;
  justify-content: start;
  align-items: start;
`;

const InnerWrapper = styled.div`
  width: 100%;
  height: 92%;
  display: flex;
  flex-direction: row;
  justify-content: start;
  align-items: start;
`;

const LeftWrapper = styled.div`
  width: 1500px;
  height: 100%;
  padding: 30px 0px 0px 30px;

  display: flex;
  flex-direction: column;
`;

const RightWrapper = styled.div`
  width: 100%;
  height: 100%;
  padding: 30px 30px 0px 15px;

  display: flex;
  flex-direction: column;

  > div {
    width: 100%;
    height: 100%;
  }
`;

const ListWrapper = styled.div`
  width: 100%;
`;

const TagTabHeader = styled.div`
  font-size: 28px;
  padding: 4px;
  border-bottom: 1px solid black;

  display: grid;
  grid-template-columns: 8fr 2fr;
  
  > div:first-child {
    display: flex;
    justify-content: space-between;
  }

  > div:last-child {
    display: flex;
    justify-content: center;
  }
`;

const TagTabName = styled.span<IPropsActive>`
  color: ${props => ((props.active === 'true') ? 'var(--ls-blue)' : 'var(--ls-gray)')};
  cursor: ${props => ((props.active === 'true') ? 'default' : 'pointer')};
  &:hover { 
    ${({ active }) => active === 'false' && `color: var(--ls-green)`}
  }
`;

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

const TagList = styled.div`
  margin-top: 10px;
  margin-bottom: 20px;

  position: absolute;
  top: 0px;

  transition-property: all;
  transition-duration: 250ms;
  transition-delay: 0s;
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