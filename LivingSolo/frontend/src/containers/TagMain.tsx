/* eslint-disable @typescript-eslint/no-explicit-any */
import React, { useEffect } from 'react';
import styled from 'styled-components';
import { useDispatch, useSelector } from 'react-redux';
import { AppDispatch } from '../store';
import { fetchTags, fetchTagsIndex, selectTag } from '../store/slices/tag';
import { TagBubbleCompact } from '../components/general/TagBubble';
import { IPropsColor } from '../utils/Interfaces';
import { getContrastYIQ } from '../styles/color';

const TagMain = () => {
  const dispatch = useDispatch<AppDispatch>();

  const { elements, index, errorState } = useSelector(selectTag);

  // Fetch Tag Related Things!
  useEffect(() => {
      dispatch(fetchTags());
      dispatch(fetchTagsIndex());
    }, [dispatch, errorState]);

  return (
    <Wrapper>
      <InnerWrapper>
        <LeftWrapper>
          <TagCategoryListWrapper>
            <TagCategoryHeader>
              <span>TagCategory</span>
            </TagCategoryHeader>
            <TagCategoryList>
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
          </TagCategoryListWrapper>
          <TagListWrapper>
            <TagHeader>
              <span>Tag</span>
            </TagHeader>
            <TagList>
              {index.map((tag) => {
                  return <TagBubbleCompact color={tag.color} key={tag.id}>{tag.name}</TagBubbleCompact>
              })}
            </TagList>         
          </TagListWrapper>
          <TagPresetListWrapper>
            <TagPresetHeader>
              <span>TagPreset</span>
            </TagPresetHeader>
            <TagPresetList>

            </TagPresetList>         
          </TagPresetListWrapper>
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

const TagCategoryListWrapper = styled.div`
  width: 100%;
`;

const TagCategoryHeader = styled.div`
  font-size: 28px;
  padding: 4px;
  border-bottom: 1px solid black;
`;

const TagCategoryList = styled.div`
  margin-bottom: 20px;
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

const TagListWrapper = styled.div`
  width: 100%;
  margin-bottom: 20px;
`;

const TagHeader = styled.div`
  font-size: 28px;
  padding: 4px;
  border-bottom: 1px solid black;
  margin-bottom: 10px;
`;

const TagList = styled.div`
  > button {
    margin: 3px 0px 0px 5px;
  }
`;

const TagPresetListWrapper = styled.div`
  width: 100%;
  margin-bottom: 20px;
`;

const TagPresetHeader = styled.div`
  font-size: 28px;
  padding: 4px;
  border-bottom: 1px solid black;
  margin-bottom: 10px;
`;

const TagPresetList = styled.div`
  > button {
    margin: 3px 0px 0px 5px;
  }
`;