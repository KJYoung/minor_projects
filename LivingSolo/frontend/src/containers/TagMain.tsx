/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useEffect, useState } from 'react';
import styled from 'styled-components';
import { useDispatch, useSelector } from 'react-redux';
import { AppDispatch } from '../store';
import { fetchTags, fetchTagsIndex, selectTag } from '../store/slices/tag';
import { TagBubbleCompact } from '../components/general/TagBubble';
import { IPropsColor } from '../utils/Interfaces';
import { getContrastYIQ } from '../styles/color';
import { CondRendAnimState, defaultCondRendAnimState, toggleCondRendAnimState } from '../utils/Rendering';
import { TagAdder, TagEditor } from '../components/Tag/TagAdder';

enum TagViewerMode {
  TagCategory, Tag, TagPreset
};

const TagMain = () => {
  const [addMode, setAddMode] = useState<CondRendAnimState>(defaultCondRendAnimState);
  const [tagViewerMode, setTagViewerMode] = useState<TagViewerMode>(TagViewerMode.TagCategory);
  const dispatch = useDispatch<AppDispatch>();

  const { elements, index, errorState } = useSelector(selectTag);

  // Fetch Tag Related Things!
  useEffect(() => {
      dispatch(fetchTags());
      dispatch(fetchTagsIndex());
    }, [dispatch, errorState]);

  const categoryEditHandler = () => {

  }

  const categoryAddToggleHandler = () => {
    toggleCondRendAnimState(addMode, setAddMode); // ON
};

  return (
    <Wrapper>
      <InnerWrapper>
        <LeftWrapper>
          <ListWrapper>
            <TagCategoryHeader>
              <span onClick={() => setTagViewerMode(TagViewerMode.TagCategory)}>TagCategory</span>
              <span onClick={() => setTagViewerMode(TagViewerMode.Tag)}>Tag</span>
              <span onClick={() => setTagViewerMode(TagViewerMode.TagPreset)}>TagPreset</span>
              <span onClick={categoryAddToggleHandler}>+</span>
            </TagCategoryHeader>
            {
              tagViewerMode === TagViewerMode.TagCategory && <TagCategoryListPosition>
                {addMode.showElem && ( true ? 
                    (<TagAdder addMode={addMode} setAddMode={setAddMode} />)
              :
                    (elements[0] && <TagEditor addMode={addMode} setAddMode={setAddMode} editObj={elements[0]} editCompleteHandler={categoryEditHandler}/>)
                )}
                <TagCategoryList style={addMode.showElem && addMode.isMounted ? { transform: "translateY(125px)" } : { transform: "translateY(0px)" }}>
                  {elements.map((tagClass) => {
                      return <TagCategoryListElement key={tagClass.id}>
                          <TagCategoryListElementHeader color={tagClass.color}>{tagClass.name}</TagCategoryListElementHeader>
                          <div>
                              {tagClass.tags?.map((tag) => {
                                  return <TagBubbleCompact color={tag.color} key={tag.id}>{tag.name}</TagBubbleCompact>
                              })}
                          </div>
                      </TagCategoryListElement>
                  })}
                </TagCategoryList>         
              </TagCategoryListPosition>
            }
            {
              tagViewerMode === TagViewerMode.Tag && <TagCategoryListPosition>
                {addMode.showElem && ( true ? 
                    (<TagAdder addMode={addMode} setAddMode={setAddMode} />)
              :
                    (elements[0] && <TagEditor addMode={addMode} setAddMode={setAddMode} editObj={elements[0]} editCompleteHandler={categoryEditHandler}/>)
                )}
                <TagCategoryList style={addMode.showElem && addMode.isMounted ? { transform: "translateY(125px)" } : { transform: "translateY(0px)" }}>
                    {index.map((tag) => {
                        return <TagBubbleCompact color={tag.color} key={tag.id}>{tag.name}</TagBubbleCompact>
                    })}     
                </TagCategoryList>         
              </TagCategoryListPosition>
            }
            {
              tagViewerMode === TagViewerMode.TagPreset && <TagCategoryListPosition>
                {addMode.showElem && ( true ? 
                    (<TagAdder addMode={addMode} setAddMode={setAddMode} />)
              :
                    (elements[0] && <TagEditor addMode={addMode} setAddMode={setAddMode} editObj={elements[0]} editCompleteHandler={categoryEditHandler}/>)
                )}
                <TagCategoryList style={addMode.showElem && addMode.isMounted ? { transform: "translateY(125px)" } : { transform: "translateY(0px)" }}>
                  Preset
                </TagCategoryList>         
              </TagCategoryListPosition>
            }
          </ListWrapper>
        </LeftWrapper>
        <RightWrapper></RightWrapper>
        
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

const TagCategoryHeader = styled.div`
  font-size: 28px;
  padding: 4px;
  border-bottom: 1px solid black;
`;

const TagCategoryListPosition = styled.div`
  width: 100%;
  position: relative;
`;

const TagCategoryList = styled.div`
  margin-bottom: 20px;

  position: absolute;
  top: 0px;

  transition-property: all;
  transition-duration: 250ms;
  transition-delay: 0s;
`;

const TagCategoryListElement = styled.div`
  margin-bottom: 10px;
  padding-bottom: 10px;
  padding-left: 10px;
  border-bottom: 1px solid black;

  &:first-child{
    padding-top: 10px;
  }
`;

const TagCategoryListElementHeader = styled.div<IPropsColor>`
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