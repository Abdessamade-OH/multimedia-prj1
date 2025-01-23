import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DescriptorVisualizationComponent } from './descriptor-visualization.component';

describe('DescriptorVisualizationComponent', () => {
  let component: DescriptorVisualizationComponent;
  let fixture: ComponentFixture<DescriptorVisualizationComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [DescriptorVisualizationComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(DescriptorVisualizationComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
